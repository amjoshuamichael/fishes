pub mod vk {
    #![allow(unused_imports, ambiguous_glob_reexports)]
    pub use vulkano::*;
    pub use pipeline::*;
    pub use pipeline::layout::*;
    pub use pipeline::graphics::{
        *, vertex_input::*, input_assembly::*, viewport::*, rasterization::*, multisample::*,
        color_blend::*, subpass::*,
    };
    pub use buffer::*;
    pub use instance::*;
    pub use image::{*, view::ImageView, sampler::*};
    pub use memory::*;
    pub use device::*;
    pub use device::physical::PhysicalDevice as GPU;
    pub use device::physical::PhysicalDeviceType as GPUKind;
    pub use format::*;
    pub use memory::allocator::*;
    pub use ordered_passes_renderpass as create_render_pass;
    pub use render_pass::*;
    pub use command_buffer::{*, allocator::*, sys::*};
    pub use sync::future::*;
    pub use shader::*;
    pub use command_buffer::*;
    pub use descriptor_set::{*, allocator::*, layout::*};
}
pub use vk::{VertexDefinition, Vertex, GpuFuture, PrimaryCommandBufferAbstract, Pipeline};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

mod text;

use std::sync::Arc;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;
const VIEWPORT: vk::Viewport = vk::Viewport {
    offset: [0.0, 0.0],
    extent: [WIDTH as f32, HEIGHT as f32],
    depth_range: 0.0..=1.0,
};

const IMAGE_FILE: &'static str = "./earth.png";
const TEST_SLOT_COUNT: usize = 400;

type Color = [u8; 4];

fn main() {
    let reference_bytes = png_read(IMAGE_FILE);

    let app = App::new();
        
    
    let test_slots = (0..TEST_SLOT_COUNT).into_iter()
        .map(|_| TestSlot::new(&app.memory_allocator, &app.render_pass))
        .collect::<Vec<_>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut item_configuration = if let Ok(json) = std::fs::read_to_string("best.json") {
        serde_json::from_str(&*json).unwrap()
    } else {
        println!("initializing a new config...");
        ItemConfiguration::default()
    };

    let mut fontr  = text::load_fonts(&app);

    let mut best_score = std::u64::MAX;
    let mut best_index = 0;
    for n in 0.. {
        let Some(mutation) = mutate(&item_configuration, &test_slots, &mut rng, &app, &mut fontr, &reference_bytes, best_score)
            else { continue };
        println!("score: {}, via: {}, count: {}", mutation.score, mutation.new_config.got_here_via, mutation.new_config.texts.len());
        item_configuration = mutation.new_config;

        if n % 10 == 0 {
            let best_json = serde_json::to_string(&item_configuration).unwrap();
            use std::{fs::*, io::*};
            File::create("best.json").expect("unable to create json file!")
                .write_all(best_json.as_bytes()).unwrap();
            png_write(test_slots[best_index].as_bytes(), "best.png");
            println!("wrote!");
        }

        best_index = mutation.best_index;
        best_score = mutation.score;
    }

    let best_json = serde_json::to_string(&item_configuration).unwrap();
    use std::{fs::*, io::*};
    File::create("best.json").expect("unable to create json file!")
        .write_all(best_json.as_bytes()).unwrap();
    png_write(test_slots[best_index].as_bytes(), "best.png");
}

struct MutationResults {
    new_config: ItemConfiguration,
    best_index: usize,
    score: u64,
}

fn mutate(
    item: &ItemConfiguration, 
    test_slots: &Vec<TestSlot>,
    rng: &mut rand::rngs::StdRng,
    app: &App,
    fontr: &mut text::FontResources,
    reference_bytes: &Box<[u8]>,
    mut best_score: u64,
) -> Option<MutationResults> {
    let mut variations = (0..test_slots.len()).into_iter()
        .map(|_| item.get_variation(rng))
        .collect::<Vec<_>>();

    draw_variations(&mut variations, app, fontr, test_slots);

    let scores = test_slots.iter()
        .map(|test_slot| cmp_bytes(test_slot.as_bytes(), &reference_bytes))
        .collect::<Vec<_>>();
    let mut best_index = None;
    for (s, score) in scores.into_iter().enumerate() {
        if score < best_score {
            best_index = Some(s);
            best_score = score;
        }
    }

    best_index.map(|best_index| MutationResults {
        new_config: variations.remove(best_index),
        best_index,
        score: best_score,
    })
}

// An unused function for 'simplifying the image' by generating various mutations that remove
// elements, and then choosing the best one. In testing, this was found to be only useful
// sometimes. Worth keeping around.
fn cut_unnescessary(
    item: &ItemConfiguration, 
    test_slots: &Vec<TestSlot>,
    rng: &mut rand::rngs::StdRng,
    app: &App,
    fontr: &mut text::FontResources,
    reference_bytes: &Box<[u8]>,
    best_score: u64,
) -> Option<ItemConfiguration> {
    let mut variations = item.simplifications(rng);

    draw_variations(&mut variations, app, fontr, test_slots);

    let scores = test_slots.iter().take(variations.len())
        .map(|test_slot| cmp_bytes(test_slot.as_bytes(), &reference_bytes))
        .collect::<Vec<_>>();
    for (s, score) in scores.into_iter().enumerate() {
        if score == best_score {
            return Some(variations.remove(s));
        }
    }

    None
}

fn draw_variations(
    variations: &mut Vec<ItemConfiguration>,
    app: &App,
    fontr: &mut text::FontResources,
    test_slots: &Vec<TestSlot>,
) {
    let mut vertex_spans = Vec::<(u32, u32)>::new();

    {
        let mut vertices = Vec::<MyVertex>::new();
        for variation in &mut *variations { 
            let span_start = vertices.len() as u32;
            variation.push_vertices(fontr, &mut vertices);
            let span_end = vertices.len() as u32;
            vertex_spans.push((span_start, span_end - span_start));
        }

        let new_vertex_buffer = vk::Buffer::from_iter(
            app.memory_allocator.clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::VERTEX_BUFFER | vk::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_HOST | 
                    vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        ).unwrap();

        let mut uploads = vk::AutoCommandBufferBuilder::primary(
            &app.command_buffer_allocator,
            app.transfer_queue.queue_family_index(),
            vk::CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        uploads.copy_buffer(vk::CopyBufferInfo::buffers(new_vertex_buffer, app.vertex_buffer.clone())).unwrap();

        uploads.build().unwrap()
            .execute(app.transfer_queue.clone()).unwrap()
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();
    }

    let mut builder = vk::AutoCommandBufferBuilder::primary(
        &app.command_buffer_allocator,
        app.graphics_queue.queue_family_index(),
        vk::CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .set_viewport(0, [VIEWPORT].into_iter().collect())
        .unwrap()
        .bind_pipeline_graphics(app.pipeline.clone())
        .unwrap()
        .bind_vertex_buffers(0, app.vertex_buffer.clone())
        .unwrap()
        .bind_descriptor_sets(vk::PipelineBindPoint::Graphics, app.pipeline.layout().clone(), 0, fontr.descriptor_set.clone())
        .unwrap();

    for (v, vertex_span) in vertex_spans.iter().enumerate() {
        builder.begin_render_pass(
            vk::RenderPassBeginInfo {
                clear_values: vec![Some([
                  0.0, 1.0, 0.0, 1.0
                ].into())],
                ..vk::RenderPassBeginInfo::framebuffer(test_slots[v].framebuffer.clone())
            },
            vk::SubpassBeginInfo {
                contents: vk::SubpassContents::Inline,
                ..Default::default()
            },
        ).unwrap()
            .draw(vertex_span.1, 1, vertex_span.0, 0).unwrap()
            .end_render_pass(Default::default()).unwrap();
    }

    builder.build().unwrap()
        .execute(app.graphics_queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let mut downloads = vk::AutoCommandBufferBuilder::primary(
        &app.command_buffer_allocator,
        app.transfer_queue.queue_family_index(),
        vk::CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    for test_slot in test_slots.iter().take(variations.len()) {
        downloads.copy_image_to_buffer(vk::CopyImageToBufferInfo::image_buffer(
            test_slot.image.clone(), 
            test_slot.buffer.clone()
        )).unwrap();
    }

    downloads.build().unwrap()
        .execute(app.transfer_queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

}

// A test slot is one slot for genetic mutation, rendering, and comparison.
struct TestSlot {
    image: Arc<vk::Image>,
    framebuffer: Arc<vk::Framebuffer>,
    buffer: vk::Subbuffer<[Color]>,
}

impl TestSlot {
    fn new(
        memory_allocator: &Arc<vk::StandardMemoryAllocator>,
        render_pass: &Arc<vk::RenderPass>,
    ) -> TestSlot {
        let image = vk::Image::new(
            memory_allocator.clone(), 
            vk::ImageCreateInfo {
                image_type: vk::ImageType::Dim2d,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: [WIDTH as u32, HEIGHT as u32, 1],
                usage: vk::ImageUsage::COLOR_ATTACHMENT | vk::ImageUsage::TRANSFER_SRC,
                initial_layout: vk::ImageLayout::Undefined,
                ..Default::default()
            },
            vk::AllocationCreateInfo::default(),
        ).unwrap();

        let image_view = vk::ImageView::new_default(image.clone()).unwrap();

        let buffer = vk::Buffer::new_slice::<Color>(
            memory_allocator.clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_HOST
                    | vk::MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (WIDTH * HEIGHT) as u64,
        ).unwrap();

        let framebuffer = vk::Framebuffer::new(
            render_pass.clone(),
            vk::FramebufferCreateInfo {
                attachments: vec![image_view.clone()],
                ..Default::default()
            }
        ).unwrap();

        Self { image, buffer, framebuffer }
    }

    fn as_bytes(&self) -> &[u8] {
        let nonnullptr = self.buffer.mapped_slice().unwrap();
        unsafe { std::slice::from_raw_parts(nonnullptr.cast::<u8>().as_mut(), (WIDTH * HEIGHT * 4) as usize) }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct ItemConfiguration {
    texts: Vec<Text>,
    bg_color: [f32; 3],
    got_here_via: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Text {
    pos: [f32; 2],
    color: [f32; 4],
    size: f32,
    string: String,
}

impl ItemConfiguration {
    fn get_variation(&self, rng: &mut rand::rngs::StdRng) -> Self {
        fn rand_size(rng: &mut rand::rngs::StdRng) -> f32 {
            let val = rng.gen_range(0.0..1.0);
            1.0 - val * val
        }

        use rand::*;

        // This is the place where the various mutations are applied. These have been messed with
        // and tuned to taste, and are typically changed for each image generation. Have fun!
        let mut new = self.clone();
        for _ in 0..2 {
            let rand_index = (rng.next_u32() as usize) % new.texts.len();
            let got = rng.next_u32() % 29;
            new.got_here_via = got;
            match got {
                0..3 => {
                    let c = &mut new.texts[rand_index];
                    let num_to_modify = &mut c.pos[rng.next_u32() as usize % 2];

                    let val = rng.gen_range(-0.6..0.6);
                    *num_to_modify += val * val;
                         if *num_to_modify < -1.0 { *num_to_modify = -1.0; }
                    else if *num_to_modify >  1.0 { *num_to_modify =  1.0; }
                },
                3..7 => {
                    let c = &mut new.texts[rand_index];
                    let num_to_modify = &mut c.color[rng.next_u32() as usize % 3];

                    *num_to_modify += rng.gen_range(-0.3..0.3);
                         if *num_to_modify < -1.0 { *num_to_modify = -1.0; }
                    else if *num_to_modify >  1.0 { *num_to_modify =  1.0; }
                },
                7 => {
                    let c = &mut new.texts[rand_index];
                    c.size = rand_size(rng);
                },
                8 => {
                    if new.texts.len() > 200 { continue }
                    new.texts.push(Text {
                        pos: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)],
                        color: [rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>(), 1.0],
                        size: rand_size(rng),
                        string: "your".into(),
                    });
                },
                9 => {
                    new.bg_color = [
                        rng.gen_range(0.9..1.0), 
                        rng.gen_range(0.9..1.0),
                        rng.gen_range(0.9..1.0),
                    ];
                },
                10 => {
                    if new.texts.len() > 200 { continue }
                    new.texts.insert(0, Text {
                        pos: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)],
                        color: [rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>(), 1.0],
                        size: rand_size(rng),
                        string: "eyes".into(),
                    });
                },
                11 => {
                    if new.texts.len() > 200 { continue }
                    new.texts.insert(0, Text {
                        pos: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)],
                        color: [rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>(), 1.0],
                        size: rand_size(rng),
                        string: "your".into(),
                    });
                },
                12..20 => {
                    if new.texts.len() > 1 {
                        new.texts.remove(rand_index);
                    }
                    new.texts.insert(0, Text {
                        pos: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)],
                        color: [rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>(), 1.0],
                        size: rand_size(rng),
                        string: "eyes".into(),
                    });
                },
                20..30 => {
                    if new.texts.len() > 1 {
                        let moving = new.texts.remove(rand_index);
                        //new.texts.push(moving);
                    }
                },
                _ => unreachable!(),
            }
        }

        new
    }

    fn default() -> Self {
        Self {
            texts: vec![
                Text { 
                    pos: [0.0,0.0], 
                    color: [1.0,0.0,0.0,1.0], 
                    size: 1.0, 
                    string: "eyes".into(),
                }
            ],
            bg_color: [0., 0., 0.],
            got_here_via: 100,
        }
    }

    fn push_vertices(&mut self, fontr: &mut text::FontResources, vertices: &mut Vec<MyVertex>) {
        for text in &self.texts {
            text::render_glyphs(&text.string, text.pos, text.size, text.color, vertices, fontr);
        }
    }

    fn simplifications(&self, _: &mut rand::rngs::StdRng) -> Vec<Self> {
        let mut outputs = Vec::<Self>::new();

        for c in 0..self.texts.len() {
            let mut output = self.clone();
            output.texts.remove(c);
            outputs.push(output);
        }

        outputs
    }
}

// General initialization items for Vulkan. Typical Vulkan step is done in `App::new`
struct App {
    device: Arc<vk::Device>,
    graphics_queue: Arc<vk::Queue>,
    transfer_queue: Arc<vk::Queue>,
    command_buffer_allocator: Arc<vk::StandardCommandBufferAllocator>,
    memory_allocator: Arc<vk::StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<vk::StandardDescriptorSetAllocator>,
    vertex_buffer: vk::Subbuffer<[MyVertex]>,
    render_pass: Arc<vk::RenderPass>,
    pipeline: Arc<vk::GraphicsPipeline>,
}

const REQUIRED_EXTENSIONS: vk::DeviceExtensions = vk::DeviceExtensions {
    ..vk::DeviceExtensions::empty()
};

impl App {
    fn new() -> Self {
        let library = vk::VulkanLibrary::new().unwrap();

        let instance = vk::Instance::new(
            library,
            vk::InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        let (device, graphics_queue, transfer_queue) = get_device_handles(&instance);

        let memory_allocator = Arc::new(vk::StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(vk::StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let descriptor_set_allocator = Arc::new(vk::StandardDescriptorSetAllocator::new(
            device.clone(),
            vk::StandardDescriptorSetAllocatorCreateInfo {
                set_count: 10,
                ..Default::default()
            },
        ));

        let vertex_buffer = vk::Buffer::new_slice::<MyVertex>(
            memory_allocator.clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::VERTEX_BUFFER | vk::BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            100000000,
        ).unwrap();

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: vk::Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let pipeline = make_pipeline(&device, &render_pass);

        App {
            device,
            graphics_queue,
            transfer_queue,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,
            vertex_buffer,
            render_pass,
            pipeline,
        }
    }
}

pub fn get_device_handles(
    instance: &Arc<vk::Instance>, 
) -> (Arc<vk::Device>, Arc<vk::Queue>, Arc<vk::Queue>) {
    let available_gpus = instance.enumerate_physical_devices().unwrap();

    struct BestDeviceInfo {
        gpu: Arc<vk::GPU>,
        score: usize,
        graphics_queue_index: usize,
        transfer_queue_index: usize,
    }
    let mut best_gpu: Option<BestDeviceInfo> = None;

    for gpu in available_gpus {
        if !gpu.supported_extensions().contains(&REQUIRED_EXTENSIONS) { continue }

        let mut found_graphics_queue_index = None;
        let mut found_transfer_queue_index = None;

        for (queue_index, queue) in gpu.queue_family_properties().iter().enumerate() {
            if found_graphics_queue_index.is_some() {
                if queue.queue_flags.intersects(vk::QueueFlags::TRANSFER) {
                    found_transfer_queue_index = Some(queue_index);
                }
            } else if queue.queue_flags.intersects(vk::QueueFlags::GRAPHICS) {
                found_graphics_queue_index = Some(queue_index);
            }
        }

        let Some(graphics_queue_index) = found_graphics_queue_index else { continue };
        let Some(transfer_queue_index) = found_transfer_queue_index else { continue };

        let gpu_kind_score = match gpu.properties().device_type {
            vk::GPUKind::DiscreteGpu => 6,
            vk::GPUKind::IntegratedGpu => 5,
            vk::GPUKind::VirtualGpu => 4,
            vk::GPUKind::Cpu => 3,
            vk::GPUKind::Other => 2,
            _ => 1,
        };
        let best_gpu_kind_score = if let Some(best) = &best_gpu { best.score } else { 0 };

        if best_gpu_kind_score < gpu_kind_score {
            best_gpu = Some(BestDeviceInfo {
                gpu,
                score: gpu_kind_score,
                graphics_queue_index,
                transfer_queue_index,
            });
        }
    }

    let Some(best_gpu) = best_gpu else { panic!("Your GPU is not supported!") };

    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        best_gpu.gpu.properties().device_name,
        best_gpu.gpu.properties().device_type,
    );

    let (device, mut queues) = vk::Device::new(
        best_gpu.gpu.clone(),
        vk::DeviceCreateInfo {
            enabled_extensions: REQUIRED_EXTENSIONS,
            queue_create_infos: vec![
                vk::QueueCreateInfo {
                    queue_family_index: best_gpu.graphics_queue_index as u32,
                    ..Default::default()
                },
                vk::QueueCreateInfo {
                    queue_family_index: best_gpu.transfer_queue_index as u32,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    )
    .unwrap();

    let graphics_queue = queues.next().unwrap();
    let transfer_queue = queues.next().unwrap();

    (device, graphics_queue, transfer_queue)
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(vk::Vertex, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec2 uv;
            layout(location = 2) in vec4 color;

            layout(location = 0) out vec2 out_uv;
            layout(location = 1) out vec4 out_color;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                out_uv = uv;
                out_color = color;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(set = 0, binding = 0) uniform sampler s;
            layout(set = 0, binding = 1) uniform texture2D tex;

            layout(location = 0) in vec2 in_uv;
            layout(location = 1) in vec4 in_color;

            layout(location = 0) out vec4 out_color;

            void main() {
                out_color = in_color;
                out_color.a *= texture(sampler2D(tex, s), in_uv).r;
            }
        ",
    }
}

fn make_pipeline(device: &Arc<vk::Device>, render_pass: &Arc<vk::RenderPass>) 
  -> Arc<vk::GraphicsPipeline> {
    let vs = vs::load(device.clone()).unwrap().entry_point("main").unwrap();
    let fs = fs::load(device.clone()).unwrap().entry_point("main").unwrap();

    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::new(vs),
        vk::PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = vk::PipelineLayout::new(
        device.clone(),
        vk::PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = vk::Subpass::from(render_pass.clone(), 0).unwrap();

    vk::GraphicsPipeline::new(
        device.clone(),
        None,
        vk::GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(vk::InputAssemblyState::default()),
            viewport_state: Some(vk::ViewportState::default()),
            rasterization_state: Some(vk::RasterizationState::default()),
            multisample_state: Some(vk::MultisampleState::default()),
            color_blend_state: Some(vk::ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                vk::ColorBlendAttachmentState {
                    blend: Some(vk::AttachmentBlend::alpha()),
                    ..Default::default()
                },
            )),
            dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..vk::GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

// Source code ~~stolen~~ adapted from the `png` crate.
fn png_write(data: &[u8], path: &str) {
    use std::path::Path;
    use std::fs::File;
    use std::io::BufWriter;

    let path = Path::new(path);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, WIDTH, HEIGHT); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::from_scaled(45455)); // 1.0 / 2.2, scaled by 100000
    encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));     // 1.0 / 2.2, unscaled, but rounded
    let source_chromaticities = png::SourceChromaticities::new(     // Using unscaled instantiation here
        (0.31270, 0.32900),
        (0.64000, 0.33000),
        (0.30000, 0.60000),
        (0.15000, 0.06000)
    );
    encoder.set_source_chromaticities(source_chromaticities);
    let mut writer = encoder.write_header().unwrap();

    let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    writer.write_image_data(&data).unwrap(); // Save
}

fn png_read(path: &str) -> Box<[u8]> {
    use std::fs::File;
    let decoder = png::Decoder::new(File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    assert_eq!(info.width, WIDTH);
    assert_eq!(info.height, HEIGHT);

    let mut output = vec![0u8; (WIDTH * HEIGHT * 4) as usize].into_boxed_slice();
    
    match info.color_type {
        png::ColorType::Rgb => {
            for n in 0..(WIDTH * HEIGHT) as usize {
                output[n * 4 + 0] = buf[n * 3 + 0];
                output[n * 4 + 1] = buf[n * 3 + 1];
                output[n * 4 + 2] = buf[n * 3 + 2];
            }
        }
        png::ColorType::Rgba => {
            for n in 0..(WIDTH * HEIGHT) as usize {
                output[n * 4 + 0] = buf[n * 4 + 0];
                output[n * 4 + 1] = buf[n * 4 + 1];
                output[n * 4 + 2] = buf[n * 4 + 2];
            }
        }
        _ => unimplemented!("reference png color type: {:?}", info.color_type),
    }

    output
}

#[cfg(target_arch = "x86_64")]
fn cmp_bytes(test: &[u8], target: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    assert!(test.len() == target.len());
    let len = test.len();
    let mut sum = 0u64;

    // Process 32 bytes at a time using AVX2.
    let mut i = 0;
    while i + 32 <= len {
        unsafe {
            let a = _mm256_loadu_si256(test.as_ptr().add(i) as *const _);
            let b = _mm256_loadu_si256(target.as_ptr().add(i) as *const _);
            let diff = _mm256_sad_epu8(a, b); // Absolute difference of 8-bit integers in each lane
            let mut temp = [0u64; 4];
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut _, diff);
            sum += temp.iter().sum::<u64>();
        }
        i += 32;
    }

    // Handle the remaining bytes.
    for j in i..len {
        sum += (test[j] as i64 - target[j] as i64).abs() as u64;
    }

    sum
}

#[cfg(target_arch = "aarch64")]
fn cmp_bytes(test: &[u8], target: &[u8]) -> u64 {
    use std::arch::aarch64::*;

    assert!(test.len() == target.len());
    let len = test.len();
    let mut sum = 0u64;

    let mut i = 0;
    while i + 16 <= len {
        unsafe {
            let a: uint8x16_t = vld1q_u8(test.as_ptr().add(i));
            let b: uint8x16_t = vld1q_u8(target.as_ptr().add(i));
            let diff = vabdq_u8(a, b); // absolute difference of 8-bit integers
            
            // sum across lanes
            let sum_vec = vpaddlq_u8(diff);     // pairwise sum to u16
            let sum_vec = vpaddlq_u16(sum_vec); // pairwise sum to u32
            let sum_vec = vpaddlq_u32(sum_vec); // pairwise sum to u64

            // sum to final
            sum += vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);
        }
        i += 16;
    }

    // handle remaining bytes 
    for j in i..len {
        sum += (test[j] as i64 - target[j] as i64).abs() as u64;
    }

    sum
}
