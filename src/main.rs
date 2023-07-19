use std::{
    f32::consts::PI,
    ops::{Div, Sub},
    vec, time::Duration,
};

use bevy::{
    input::mouse::MouseWheel, prelude::*, render::render_resource::PrimitiveTopology,
    sprite::MaterialMesh2dBundle, time::common_conditions::on_timer,
};
use bevy_egui::{egui, EguiContexts};
use bevy_prototype_debug_lines::*;
use rand::prelude::*;

use bevy_inspector_egui::quick::ResourceInspectorPlugin;
use bevy_inspector_egui::quick::WorldInspectorPlugin;

const tick_rate: f64 = 1./6000.;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(DebugLinesPlugin::default())
        //.add_plugin(EguiPlugin)
        .init_resource::<UiState>()
        .register_type::<UiState>()
        .add_plugins(ResourceInspectorPlugin::<UiState>::default())
        
        .init_resource::<TPS>()
        .register_type::<TPS>()
        .add_plugins(ResourceInspectorPlugin::<TPS>::default())

        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup)
        //.add_systems(Update, (go_forward.run_if(on_timer(Duration::from_secs_f64(1.0 / 60.))), update_sight.run_if(on_timer(Duration::from_secs_f64(1.0 / 60.))), go_align.run_if(on_timer(Duration::from_secs_f64(1.0 / 60.)))).chain())
        .add_systems(Update, (go_forward, update_sight, go_align).run_if(on_timer(Duration::from_secs_f64(tick_rate))).chain())
        .add_systems(Update, (go_wobble.run_if(on_timer(Duration::from_secs_f64(tick_rate))), scroll_events, update_tps))
        //.add_system(ui_example_system)
        .run();
}

#[derive(Reflect, Resource)]
struct UiState {
    triangle_size: f32,
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
            triangle_size: 5.,
            num_boids: 3000,
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
    /*
           triangle_size: 5.,
           num_boids: 5000,
           align_factor: 10.,
           coherence_factor: 0.1,
           bound_range: 70.,
           repel_distance: 1000.,
           avoidance_factor: 1.5,
           wall_range: 150.,
           turn_rate: 0.01,
           wall_factor: 2.,
           boid_speed: 0.5,
    */
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
    for ev in scroll_evr.iter() {
        match ev.unit {
            MouseScrollUnit::Line => {
                println!(
                    "Scroll (line units): vertical: {}, horizontal: {}",
                    ev.y, ev.x
                );
                let mut projection = camera_transform.single_mut();
                projection.scale *= 1.025 + ev.y * 0.225;
                projection.scale = projection.scale.max(0.25);
            }
            MouseScrollUnit::Pixel => {
                println!(
                    "Scroll (pixel units): vertical: {}, horizontal: {}",
                    ev.y, ev.x
                );
            }
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    windows: Query<&Window>,
    ui_state: Res<UiState>,
) {
    commands.spawn(Camera2dBundle::default());

    let res = &windows.single().resolution;

    for _ in 0..ui_state.num_boids {
        let mut test = Mesh::new(PrimitiveTopology::TriangleList);
        let v_pos = vec![
            [-ui_state.triangle_size, -ui_state.triangle_size / 2., 0.0],
            [ui_state.triangle_size, 0., 0.0],
            [-ui_state.triangle_size, ui_state.triangle_size / 2., 0.0],
        ];

        test.insert_attribute(Mesh::ATTRIBUTE_POSITION, v_pos);

        let rand_transform = Transform::from_translation(Vec3::new(
            (random::<f32>() - 0.5) * res.width(),
            (random::<f32>() - 0.5) * res.height(),
            0.,
        ))
        .with_rotation(Quat::from_rotation_z(random::<f32>() * 2. * PI));

        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes.add(test).into(),
                material: materials.add(ColorMaterial::from(Color::PURPLE)),
                transform: rand_transform,
                ..default()
            },
            Boid,
            Forward,
            Wobble,
            Align,
            Sight(vec![]),
        ));
    }
}

fn go_forward(mut query: Query<&mut Transform, With<Forward>>, ui_state: Res<UiState>) {
    for mut transform in query.iter_mut() {
        transform.translation = transform.translation
            + transform.rotation.mul_vec3(Vec3 {
                x: ui_state.boid_speed,
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
    mut lines: ResMut<DebugLines>,
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

        //lines.line_gradient(boid.translation, boid.translation + (alignment_target * 50.), 0., Color::BLUE, Color::WHITE);

        //Coherence
        let coherence_target = sight
            .0
            .iter()
            .map(|tranform| tranform.translation)
            .sum::<Vec3>()
            .div(sight.0.len().max(1) as f32)
            .sub(boid.translation);

        //lines.line_gradient(boid.translation, boid.translation + (coherence_target), 0., Color::GREEN, Color::WHITE);

        //Avoidance
        let avoidance_target_working = sight
            .0
            .iter()
            .filter_map(|transform| {
                let d = transform.translation.distance_squared(boid.translation);
                if d < ui_state.repel_distance && d > 0.1 {
                    Some((transform, d))
                } else {
                    None
                }
            })
            .map(|(tranform, distance)| {
                (boid.translation - tranform.translation) * (ui_state.repel_distance / distance)
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

        let window_bounds = Bounds::new(
            (res.height() / 2.) as i32,
            -(res.height() / 2.) as i32,
            -(res.width() / 2.) as i32,
            (res.width() / 2.) as i32,
        );
        draw_bounds(&mut lines, &window_bounds);

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
        /*
        lines.line_colored(
            boid.translation,
            boid.translation + (final_target.normalize()) * 50.,
            0.,
            Color::ORANGE,
        );

        lines.line_colored(
            boid.translation,
            boid.translation + boid.rotation.mul_vec3(Vec3::X) * 50.,
            0.,
            Color::BLUE,
        );

        lines.line_colored(
            boid.translation,
            boid.translation + boid.rotation.mul_vec3(Vec3::Y) * 50.,
            0.,
            Color::GREEN,
        );

        lines.line_colored(
            boid.translation,
            boid.translation + wall_force,
            0.,
            Color::GREEN,
        );
         */

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

fn update_sight(
    mut new_q: Query<(&mut Transform, &mut Sight, Entity)>,
    mut lines: ResMut<DebugLines>,
    windows: Query<&Window>,
    ui_state: Res<UiState>,
) {
    let res = &windows.single().resolution;

    let mut p = GridPartion::new(
        6,
        Bounds::new(
            res.height() as i32 / 2,
            -res.height() as i32 / 2,
            -res.width() as i32 / 2,
            res.width() as i32 / 2,
        ),
    );

    for (a, _b, c) in new_q.iter() {
        p.insert(c, a.translation)
    }

    let boxxed = GridNode::Grid(Box::from(p));
    
    let s_map = new_q
        .iter()
        .map(|(boid, _s, e)| {
            (
                e,
                query_grid(
                    &boxxed,
                    Bounds::new(
                        (boid.translation.y + ui_state.bound_range / 2.) as i32,
                        (boid.translation.y - ui_state.bound_range / 2.) as i32,
                        (boid.translation.x - ui_state.bound_range / 2.) as i32,
                        (boid.translation.x + ui_state.bound_range / 2.) as i32,
                    ),
                    &mut lines,
                ),
            )
        })
        .collect::<Vec<_>>();
    /*
    for (a, _, _) in new_q.iter() {
        let _bounds = Bounds::new(
            (a.translation.y + ui_state.bound_range / 2.) as i32,
            (a.translation.y - ui_state.bound_range / 2.) as i32,
            (a.translation.x - ui_state.bound_range / 2.) as i32,
            (a.translation.x + ui_state.bound_range / 2.) as i32,
        );

        //draw_bounds(&mut lines, &bounds);
    }
     */
    /*
    let debug_map = s_map.iter().map(|(e, v)| (new_q.get_component::<Transform>(*e).unwrap(), v.iter().map(|e| new_q.get_component::<Transform>(*e).unwrap()).collect::<Vec<_>>())).collect::<Vec<_>>();

    for (t, ts) in debug_map {
        for target in ts {
            lines.line_colored(t.translation, target.translation, 0., Color::RED)
        }
    }
     */

    if ui_state.show_gp {draw_gp(&mut lines, &boxxed);}

    for (e, v) in s_map {
        let to_up = v
            .iter()
            .map(|f| *new_q.get(*f).unwrap().0)
            .collect::<Vec<_>>();

        let (_, mut s, _) = new_q.get_mut(e).unwrap();

        s.0 = to_up;
    }
    /*

    let bound_list = new_q.iter().map(|(boid, s, e)| (Bounds::new(
        (boid.translation.y + BOUND_RANGE/2.) as i32,
        (boid.translation.y - BOUND_RANGE/2.) as i32,
        (boid.translation.x - BOUND_RANGE/2.) as i32,
        (boid.translation.x - BOUND_RANGE/2.) as i32), s))
        .collect::<Vec<_>>();

    for (b, mut s) in bound_list {
        s.0 = vec![];
    }
     */
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

fn query_grid(grid_node: &GridNode, bounds: Bounds, lines: &mut ResMut<DebugLines>) -> Vec<Entity> {
    let _mid_b = Vec3::new(
        (bounds.z + bounds.w) as f32 / 2.,
        (bounds.x + bounds.y) as f32 / 2.,
        0.,
    );

    match grid_node {
        GridNode::Grid(g) => {
            
            if simple_aabb(bounds, g.bounds) {
                let mut out_buf = vec![];
                out_buf.append(&mut query_grid(&g.top_left, bounds, lines));
                out_buf.append(&mut query_grid(&g.top_right, bounds, lines));
                out_buf.append(&mut query_grid(&g.bottom_left, bounds, lines));
                out_buf.append(&mut query_grid(&g.bottom_right, bounds, lines));
                return out_buf;
            } else {
                return vec![]
            }
             

            let test = less_simple_aabb(g.bounds, bounds);
            if test == 1 {
                let mut out_buf = vec![];
                out_buf.append(&mut query_grid(&g.top_left, bounds, lines));
                out_buf.append(&mut query_grid(&g.top_right, bounds, lines));
                out_buf.append(&mut query_grid(&g.bottom_left, bounds, lines));
                out_buf.append(&mut query_grid(&g.bottom_right, bounds, lines));
                return out_buf;
            } else if test == 2 {
                return take_all(grid_node);
            } else {
                return vec![]
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
        if b1.x <= b2.x && b1.y >= b2.x && b1.y >= b2.y && b1.w <= b2.w {2}
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
pub struct Sight(Vec<Transform>);

#[derive(Component)]
pub struct Boid;

enum GridNode {
    Grid(Box<GridPartion>),
    Leaf(Bounds, Vec<Entity>),
    Empty(Bounds, u8),
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
