use std::{
    f32::consts::PI,
    ops::{Div, Sub},
    time::Duration, sync::Arc, rc::Rc,
};

use rayon::prelude::*;

use bevy::{
    input::mouse::MouseWheel, prelude::*, render::render_resource::PrimitiveTopology,
    sprite::MaterialMesh2dBundle, time::common_conditions::on_timer,
};

use bevy_prototype_debug_lines::*;
use rand::prelude::*;

use bevy_inspector_egui::quick::ResourceInspectorPlugin;

const TICK_RATE: f64 = 1./60.;
const SCALE_FACTOR: f32 = 0.4;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(DebugLinesPlugin::default())

        .init_resource::<UiState>()
        .register_type::<UiState>()
        .add_plugins(ResourceInspectorPlugin::<UiState>::default())
        
        .init_resource::<TPS>()
        .register_type::<TPS>()
        .add_plugins(ResourceInspectorPlugin::<TPS>::default())

        .init_resource::<boid_resources>()

        .init_resource::<GridRes>()
        
        //.add_plugins(WorldInspectorPlugin::new())

        .add_systems(Startup, setup)
        
        .add_systems(Update, (gp_draw_check))
        .add_systems(Update, ((population_manager, update_grid).chain()).run_if(on_timer(Duration::from_secs_f64(1./5.))))
        .add_systems(Update, (go_forward, go_align, (update_sight).chain()).run_if(on_timer(Duration::from_secs_f64(TICK_RATE))).chain())
        .add_systems(Update, (go_wobble.run_if(on_timer(Duration::from_secs_f64(TICK_RATE))), scroll_events, update_tps))
         
        .run();
}

const TRIANGLE_SIZE: f32 = 5. * SCALE_FACTOR;

#[derive(Reflect, Resource)]
struct UiState {
    num_boids: i32,
    align_factor: f32,
    coherence_factor: f32,
    bound_range: f32,
    repel_distance: f32,
    avoidance_factor: f32,
    wall_range: f32,
    turn_rate: f32,
    wall_factor: f32,
    boid_speed: f32,
    grid_update_rate: f64,
    show_gp: bool,
}

#[derive(Reflect, Resource)]
struct TPS {
    tps: f32,
    average_tps: f32,
    count: f32,
}

impl Default for TPS {
    fn default() -> Self {
        Self { tps: 0., average_tps: 0., count: 1. }
    }
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            num_boids: 10000,
            align_factor: 15.,
            coherence_factor: 0.1,
            bound_range: 50.,
            repel_distance: 100.,
            avoidance_factor: 20.,
            wall_range: 150.,
            turn_rate: 0.01,
            wall_factor: 20.,
            boid_speed: 0.5,
            grid_update_rate: 1.0 / 60.0,
            show_gp: false,
        }
    }
}

fn update_tps(
    time: Res<Time>,
    mut tps: ResMut<TPS>,
) {
    tps.tps = (1./time.delta_seconds_f64()) as f32;
    if tps.tps.is_infinite() {return}
    tps.average_tps = tps.average_tps * (tps.count - 1.)/tps.count + tps.tps/tps.count;
    tps.count += 1.;

}

fn scroll_events(
    mut scroll_evr: EventReader<MouseWheel>,
    mut camera_transform: Query<&mut OrthographicProjection, With<Camera>>,
) {
    use bevy::input::mouse::MouseScrollUnit;
    for mw in scroll_evr.iter() {
        match mw.unit {
            MouseScrollUnit::Line => {
                let mut projection = camera_transform.single_mut();
                projection.scale *= 1.025 + mw.y * 0.225;
                projection.scale = projection.scale.max(0.25);
            }
            MouseScrollUnit::Pixel => {}
        }
    }
}

fn setup(
    mut commands: Commands,
    windows: Query<&Window>,
    ui_state: Res<UiState>,
    boid_res: Res<boid_resources>
) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 5.0),
        ..Default::default()
    });

    let res = &windows.single().resolution;

    for _ in 0..ui_state.num_boids {
        spawn_random_boid(&mut commands, &boid_res, res)
    }
}

fn spawn_random_boid(commands: &mut Commands, boid_res: &boid_resources, res: &bevy::window::WindowResolution) {
    let rand_transform = Transform::from_translation(Vec3::new(
        (random::<f32>() - 0.5) * res.width(),
        (random::<f32>() - 0.5) * res.height(),
        0.,
    ))
    .with_rotation(Quat::from_rotation_z(random::<f32>() * 2. * PI));

    commands.spawn((
        MaterialMesh2dBundle {
            mesh: boid_res.mesh.clone().into(),
            material: boid_res.material.clone(),
            transform: rand_transform,
            ..default()
        },
        Boid,
        Forward,
        Wobble,
        Align,
        Sight(Arc::from([])),
    ));
}

fn population_manager(
    mut commands: Commands, 
    boids: Query<Entity, With<Boid>>, 
    ui_state: Res<UiState>,
    windows: Query<&Window>,
    boid_res: Res<boid_resources>
) {
    let current_pop = boids.iter().count();
    let desired_pop = ui_state.num_boids as usize;

    if current_pop == desired_pop {return};

    if current_pop > desired_pop {
        //Cull boids
        boids.iter().take(current_pop - desired_pop).for_each(|e| commands.entity(e).despawn());

    } else if current_pop < desired_pop {
        for i in 0 .. desired_pop - current_pop {
            spawn_random_boid(&mut commands, &boid_res, &windows.single().resolution)
        }
    }
}

#[derive(Resource)]
struct boid_resources {
    mesh: Handle<Mesh>,
    material: Handle<ColorMaterial>,
}

impl FromWorld for boid_resources {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        let mut boid_mesh = Mesh::new(PrimitiveTopology::TriangleList);
        
        let v_pos = vec![
            [-TRIANGLE_SIZE , -TRIANGLE_SIZE / 2., 0.0],
            [TRIANGLE_SIZE, 0., 0.0],
            [-TRIANGLE_SIZE, TRIANGLE_SIZE / 2., 0.0],
        ];

        boid_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, v_pos);

        let mesh_handle = meshes.add(boid_mesh);

        let mut materials = world.resource_mut::<Assets<ColorMaterial>>();
        let material_handle = materials.add(ColorMaterial::from(Color::PURPLE));

        Self { mesh: mesh_handle, material: material_handle }
    }
}

fn go_forward(mut query: Query<&mut Transform, With<Forward>>, ui_state: Res<UiState>) {
    for mut transform in query.iter_mut() {
        transform.translation = transform.translation
            + transform.rotation.mul_vec3(Vec3 {
                x: ui_state.boid_speed * SCALE_FACTOR,
                y: 0.,
                z: 0.,
            });
    }
}

fn go_wobble(mut query: Query<&mut Transform, With<Wobble>>) {
    for mut transform in query.iter_mut() {
        transform.rotate_z((random::<f32>() - 0.5) * 0.1);
    }
}

fn go_align(
    mut query: Query<(&mut Transform, &Sight, Entity), With<Align>>,
    windows: Query<&Window>,
    ui_state: Res<UiState>,
) {
    let _boids: Vec<_> = query.iter().map(|(a, _b, _e)| a.to_owned()).collect();

    //For each boid
    for (mut boid, sight, _entity) in query.iter_mut() {
        //Alignment
        let alignment_target = sight
            .0
            .iter()
            .map(|tranform| {
                tranform.rotation.mul_vec3(Vec3 {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                })
            })
            .sum::<Vec3>()
            .normalize_or_zero();

        //Coherence
        let coherence_target = sight
            .0
            .iter()
            .map(|tranform| tranform.translation)
            .sum::<Vec3>()
            .div(sight.0.len().max(1) as f32)
            .sub(boid.translation);

        //Avoidance
        let avoidance_target_working = sight
            .0
            .iter()
            .filter_map(|transform| {
                let d = transform.translation.distance_squared(boid.translation);
                if d < ui_state.repel_distance*SCALE_FACTOR*SCALE_FACTOR && d > 0.1 {
                    Some((transform, d))
                } else {
                    None
                }
            })
            .map(|(tranform, distance)| {
                (boid.translation - tranform.translation) * (ui_state.repel_distance*SCALE_FACTOR*SCALE_FACTOR / distance)
            })
            .fold((0., Vec3::ZERO), |(a, b), x| (a + 1., b + x));

        //Make sure we're not going to div by zero
        let avoidance_target = if avoidance_target_working.0 == 0. {
            Vec3::ZERO
        } else {
            avoidance_target_working.1.div(avoidance_target_working.0)
        };

        //Wall Avoidance
        let res = &windows.single().resolution;

        let wall_force = Vec3 {
            x: -0_f32.max(boid.translation.x - (res.width() / 2. - ui_state.wall_range))
                - 0_f32.min(boid.translation.x - (-res.width() / 2. + ui_state.wall_range)),
            y: -0_f32.max(boid.translation.y - (res.height() / 2. - ui_state.wall_range))
                - 0_f32.min(boid.translation.y - (-res.height() / 2. + ui_state.wall_range)),
            z: 0.,
        };

        let final_target = (alignment_target * ui_state.align_factor
            + coherence_target * ui_state.coherence_factor
            + avoidance_target * ui_state.avoidance_factor
            + wall_force * ui_state.wall_factor)
            .normalize_or_zero();

        if final_target == Vec3::ZERO {
            println!("Warning");
            continue;
        }

        let boid_forward = boid.rotation * Vec3::X;
        let boid_left = boid.rotation * Vec3::Y;

        let dot_f = boid_forward.dot(final_target);
        let dot_l = boid_left.dot(final_target);

        //Use dot_L to get which way to turn
        //Use dot_F to tell us how much to turn

        let total_turn = dot_f.clamp(-1.0, 1.0).acos();

        boid.rotate_z(
            f32::copysign(1.0, dot_l) * total_turn.clamp(-ui_state.turn_rate, ui_state.turn_rate),
        );
    }
}

fn update_grid(
    windows: Query<&Window>,
    sight_query: Query<(&mut Transform, &mut Sight, Entity)>,
    mut grid_res: ResMut<GridRes>,
) {
    let res = &windows.single().resolution;

    let mut grid_partition = GridPartion::new(
        8,
        Bounds::new(
            res.height() as i32 / 2,
            -res.height() as i32 / 2,
            -res.width() as i32 / 2,
            res.width() as i32 / 2,
        ),
    );

    for (transform, _b, entity) in sight_query.iter() {
        grid_partition.insert(entity, transform.translation)
    }
    
    grid_res.grid = GridNode::Grid(Box::from(grid_partition));
}

fn spacial_ordering(grid: &GridNode) -> Vec<Entity> {
    match grid {
        GridNode::Grid(g) => vec![spacial_ordering(&g.top_left), spacial_ordering(&g.top_right),
                                                    spacial_ordering(&g.bottom_left), spacial_ordering(&g.bottom_right)].concat(),
        GridNode::Leaf(_, v) => return v.clone(),
        GridNode::Empty(_, _) => return vec![],
    }
}

fn update_sight(
    mut sight_query: Query<(&mut Transform, &mut Sight, Entity)>,
    ui_state: Res<UiState>,
    grid_res: ResMut<GridRes>,
) {
    if let GridNode::Grid(gp) = &grid_res.grid {
        let v = vec![&gp.top_left, &gp.top_right, &gp.bottom_left, &gp.bottom_right]
            .par_iter()
            .map(|a| spacial_ordering(a)
            .into_iter()
            .map(|e| sight_query.get(e).map(|(t, _, e)| (e, t))))
            .flatten_iter()
            .filter_map(|e| e.ok())
            .collect::<Vec<_>>();
    
        //In limited tests this seems to improve performance.
        //let v = spacial_ordering(&grid_res.grid).into_iter().map(|e| (e, sight_query.get(e).unwrap().0)).collect::<Vec<_>>();
        
        //let v: Vec<(Entity, &Transform)> = sight_query.into_iter().map(|(t, _s, e)| (e, t)).collect::<Vec<_>>();
        let sight_map = v.into_par_iter().map(|(e, boid)|
        {
            (
                e,
                query_grid(
                    &grid_res.grid,
                    Bounds::new(
                        (boid.translation.y + ui_state.bound_range*SCALE_FACTOR / 2.) as i32,
                        (boid.translation.y - ui_state.bound_range*SCALE_FACTOR / 2.) as i32,
                        (boid.translation.x - ui_state.bound_range*SCALE_FACTOR / 2.) as i32,
                        (boid.translation.x + ui_state.bound_range*SCALE_FACTOR / 2.) as i32,
                    ),
                ),
            )
        }).collect::<Rc<_>>();

        for (e, v) in sight_map.into_iter() {
            let updated_sight = v
                .iter()
                .map(|f| sight_query.get(*f).map(|e| e.0))
                .filter(|e| e.is_ok())
                .map(|e| *e.unwrap())
                .collect::<Arc<_>>();

            let (_, mut s, _) = sight_query.get_mut(*e).unwrap();

            s.0 = updated_sight;
        }
    }
}


fn gp_draw_check(
    mut lines: ResMut<DebugLines>,
    ui_state: Res<UiState>,
    grid_res: ResMut<GridRes>,
    windows: Query<&Window>,
) {
    if ui_state.show_gp {
        draw_gp(&mut lines, &grid_res.grid);
    }

    let res = &windows.single().resolution;

    let window_bounds = Bounds::new(
        (res.height() / 2.) as i32,
        -(res.height() / 2.) as i32,
        -(res.width() / 2.) as i32,
        (res.width() / 2.) as i32,
    );

    draw_bounds(&mut lines, &window_bounds);
}

fn draw_bounds(lines: &mut ResMut<DebugLines>, bounds: &Bounds) {
    lines.line_colored(
        Vec3::new(bounds.z as f32, bounds.x as f32, 0.),
        Vec3::new(bounds.w as f32, bounds.x as f32, 0.),
        0.,
        Color::GOLD,
    );
    lines.line_colored(
        Vec3::new(bounds.z as f32, bounds.y as f32, 0.),
        Vec3::new(bounds.w as f32, bounds.y as f32, 0.),
        0.,
        Color::GOLD,
    );
    lines.line_colored(
        Vec3::new(bounds.z as f32, bounds.x as f32, 0.),
        Vec3::new(bounds.z as f32, bounds.y as f32, 0.),
        0.,
        Color::GOLD,
    );
    lines.line_colored(
        Vec3::new(bounds.w as f32, bounds.x as f32, 0.),
        Vec3::new(bounds.w as f32, bounds.y as f32, 0.),
        0.,
        Color::GOLD,
    );
}

fn draw_gp(lines: &mut ResMut<DebugLines>, grid: &GridNode) {
    match grid {
        GridNode::Grid(g) => {
            draw_bounds(lines, &g.bounds);
            draw_gp(lines, &g.top_left);
            draw_gp(lines, &g.top_right);
            draw_gp(lines, &g.bottom_left);
            draw_gp(lines, &g.bottom_right);
        }
        GridNode::Leaf(b, _) => draw_bounds(lines, b),
        GridNode::Empty(_, _) => {}
    }
}

fn query_grid(grid_node: &GridNode, bounds: Bounds) -> Vec<Entity> {
    let _mid_b = Vec3::new(
        (bounds.z + bounds.w) as f32 / 2.,
        (bounds.x + bounds.y) as f32 / 2.,
        0.,
    );

    match grid_node {
        GridNode::Grid(g) => {
            if false {
                if simple_aabb(bounds, g.bounds) {
                    let mut out_buf = vec![];
                    out_buf.append(&mut query_grid(&g.top_left, bounds));
                    out_buf.append(&mut query_grid(&g.top_right, bounds));
                    out_buf.append(&mut query_grid(&g.bottom_left, bounds));
                    out_buf.append(&mut query_grid(&g.bottom_right, bounds));
                    return out_buf;
                } else {
                    return vec![]
                }
            } else {

                let test = less_simple_aabb(g.bounds, bounds);
                if test == 1 {
                    let mut out_buf = vec![];
                    out_buf.append(&mut query_grid(&g.top_left, bounds));
                    out_buf.append(&mut query_grid(&g.top_right, bounds));
                    out_buf.append(&mut query_grid(&g.bottom_left, bounds));
                    out_buf.append(&mut query_grid(&g.bottom_right, bounds));
                    return out_buf;
                } else if test == 2 {
                    return take_all(grid_node);
                } else {
                    return vec![]
                }
            }


            
        }
        GridNode::Leaf(b, v) => {
            //If the leaf is at least partially inside the bound we return all
            if simple_aabb(bounds, *b) {
                return v.clone();
            } else {
                vec![]
            }
        }
        GridNode::Empty(_, _) => vec![],
    }
}

//Do b1 and b2 overlap?
fn simple_aabb(b1: Bounds, b2: Bounds) -> bool {
    !(b1.x < b2.y || b1.y > b2.x || b1.z > b2.w || b1.w < b2.z)
}


fn less_simple_aabb(b1: Bounds, b2: Bounds) -> u8 {
    //Is there overlap?
    if !(b1.x < b2.y || b1.y > b2.x || b1.z > b2.w || b1.w < b2.z) {
        //If there is then classify
        //We will assume that both bbs are the correct way up (top is up)
        //Is b2 IN b1?
        if b1.x <= b2.x && b1.z >= b2.z && b1.y >= b2.y && b1.w <= b2.w {2}
        else {1}
    }
    else {0}
}

fn take_all(grid_node: &GridNode) -> Vec<Entity> {
    match grid_node {
        GridNode::Grid(g) => {
            let mut ret = vec![];
            ret.append(&mut take_all(&g.top_left));
            ret.append(&mut take_all(&g.top_right));
            ret.append(&mut take_all(&g.bottom_left));
            ret.append(&mut take_all(&g.bottom_right));
            ret
        }
        GridNode::Leaf(_, list) => list.clone(),
        GridNode::Empty(_, _) => vec![],
    }
}

fn bounds_teleport(mut query: Query<&mut Transform>, windows: Query<&Window>) {
    let res = &windows.single().resolution;

    for mut transform in query.iter_mut() {
        if transform.translation.y > res.height() / 2. {
            transform.translation.y -= res.height();
        }

        if transform.translation.x > res.width() / 2. {
            transform.translation.x -= res.width();
        }

        if transform.translation.y < -res.height() / 2. {
            transform.translation.y += res.height();
        }

        if transform.translation.x < -res.width() / 2. {
            transform.translation.x += res.width();
        }
    }
}

#[derive(Component)]
pub struct Forward;

#[derive(Component)]
pub struct Wobble;

#[derive(Component)]
pub struct Align;

#[derive(Component)]
pub struct Sight(Arc<[Transform]>);

#[derive(Component)]
pub struct Boid;

#[derive(Resource, Default)]
struct GridRes {
    grid: GridNode,
}


enum GridNode {
    Grid(Box<GridPartion>),
    Leaf(Bounds, Vec<Entity>),
    Empty(Bounds, u8),
}

impl Default for GridNode {
    fn default() -> Self {
        GridNode::Empty(IVec4::default(), 0)
    }
}
struct GridPartion {
    bounds: Bounds,

    top_left: GridNode,
    top_right: GridNode,
    bottom_left: GridNode,
    bottom_right: GridNode,
}

type Bounds = IVec4;
//top bottom, left, right
//x, y, z, w
impl GridPartion {
    fn new(depth: u8, bounds: Bounds) -> GridPartion {
        GridPartion {
            bounds,
            top_left: GridNode::Empty(
                Bounds::new(
                    bounds.x,
                    (bounds.x + bounds.y) / 2,
                    bounds.z,
                    (bounds.w + bounds.z) / 2,
                ),
                depth - 1,
            ),
            top_right: GridNode::Empty(
                Bounds::new(
                    bounds.x,
                    (bounds.x + bounds.y) / 2,
                    (bounds.w + bounds.z) / 2,
                    bounds.w,
                ),
                depth - 1,
            ),
            bottom_left: GridNode::Empty(
                Bounds::new(
                    (bounds.x + bounds.y) / 2,
                    bounds.y,
                    bounds.z,
                    (bounds.w + bounds.z) / 2,
                ),
                depth - 1,
            ),
            bottom_right: GridNode::Empty(
                Bounds::new(
                    (bounds.x + bounds.y) / 2,
                    bounds.y,
                    (bounds.w + bounds.z) / 2,
                    bounds.w,
                ),
                depth - 1,
            ),
        }
    }

    fn insert(&mut self, entity: Entity, translation: Vec3) {
        let midline_v = (self.bounds.x + self.bounds.y) / 2;
        let midline_h = (self.bounds.z + self.bounds.w) / 2;

        if translation.y > midline_v as f32 {
            if translation.x > midline_h as f32 {
                insert_helper(&mut self.top_right, entity, translation)
            } else {
                insert_helper(&mut self.top_left, entity, translation)
            }
        } else {
            if translation.x > midline_h as f32 {
                insert_helper(&mut self.bottom_right, entity, translation)
            } else {
                insert_helper(&mut self.bottom_left, entity, translation)
            }
        }
    }
}

fn insert_helper(node_to_insert_into: &mut GridNode, entity: Entity, translation: Vec3) {
    match node_to_insert_into {
        GridNode::Grid(ref mut g) => g.insert(entity, translation),
        GridNode::Leaf(_b, ref mut v) => v.push(entity),
        GridNode::Empty(bounds_new, depth_new) => {
            if *depth_new == 0 {
                *node_to_insert_into = GridNode::Leaf(*bounds_new, vec![entity])
            } else {
                let mut new_grid = GridPartion::new(*depth_new, *bounds_new);
                new_grid.insert(entity, translation);
                *node_to_insert_into = GridNode::Grid(Box::from(new_grid))
            }
        }
    }
}
