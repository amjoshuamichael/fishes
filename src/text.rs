const GLYPH_CACHE_SIZE: u32 = 400;

use std::sync::Arc;
use std::collections::HashMap;

use rusttype::*;
use super::vk;
use vk::{Pipeline, PrimaryCommandBufferAbstract};
use vk::sync::GpuFuture;

// Font shaping is the bottleneck in this program. You could cache shaped glyphs, but this was not
// worth the performance boost in my usage.
pub struct FontResources {
    font: Font<'static>,
    pub descriptor_set: Arc<vk::PersistentDescriptorSet>,
    cache: gpu_cache::Cache<'static>,
    pub glyphs: Vec::<PositionedGlyph<'static>>,
    pub text_render_cache: HashMap<String, Vec<MyVertex>>,
}

pub fn load_fonts(
    app: &super::App,
) -> FontResources {
    let font_bytes = include_bytes!("../Times New Roman.ttf");
    let font = Font::try_from_bytes(font_bytes).unwrap();

    let cache_tex = vk::Image::new(
        app.memory_allocator.clone(),
        vk::ImageCreateInfo {
            image_type: vk::ImageType::Dim2d,
            format: vk::Format::R8_UNORM,
            extent: [GLYPH_CACHE_SIZE, GLYPH_CACHE_SIZE, 1],
            usage: vk::ImageUsage::TRANSFER_SRC | vk::ImageUsage::TRANSFER_DST | vk::ImageUsage::SAMPLED,
            ..Default::default()
        },
        vk::AllocationCreateInfo::default(),
    )
    .unwrap();

    let cache_view = vk::ImageView::new_default(cache_tex.clone()).unwrap();

    let mut cache = rusttype::gpu_cache::Cache::builder()
        .dimensions(GLYPH_CACHE_SIZE as u32, GLYPH_CACHE_SIZE as u32)
        .scale_tolerance(0.2)
        .position_tolerance(0.2)
        .build::<'static>();

    let fonts_sampler = vk::Sampler::new(
        app.device.clone(), 
        vk::SamplerCreateInfo::simple_repeat_linear()
    ).unwrap();

    let descriptor_set = vk::PersistentDescriptorSet::new(
        &app.descriptor_set_allocator,
        app.pipeline.layout().set_layouts()[0].clone(),
        [ vk::WriteDescriptorSet::sampler(0, fonts_sampler.clone()),
          vk::WriteDescriptorSet::image_view(1, cache_view.clone()) ],
        [],
    ).unwrap();

    let glyphs = prepare_text(&font, &mut cache, &cache_tex, app);

    let text_render_cache = HashMap::new();

    FontResources { font, descriptor_set, glyphs, cache, text_render_cache }
}

pub fn prepare_text(
    font: &Font<'static>,
    cache: &mut rusttype::gpu_cache::Cache<'static>,
    cache_tex: &Arc<vk::Image>,
    app: &super::App,
) -> Vec<PositionedGlyph<'static>> {
    use rusttype::*;

    let mut glyphs = Vec::<PositionedGlyph<'static>>::new();

    for char in "abcdefghijklmnopqrstuvwxyz()".chars() {
        let glyph = font.glyph(char)
            .scaled(Scale::uniform(80.))
            .positioned(Point { x: 0., y: 0. });

        cache.queue_glyph(0, glyph.clone());

        glyphs.push(glyph);
    }

    let mut uploads = vk::AutoCommandBufferBuilder::primary(
        &app.command_buffer_allocator,
        app.transfer_queue.queue_family_index(),
        vk::CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    cache.cache_queued(|rect, data| {
        let buffer = vk::Buffer::new_slice::<u8>(
            app.memory_allocator.clone(),
            vk::BufferCreateInfo {
                usage: vk::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::PREFER_HOST
                    | vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.len() as u64,
        ).unwrap();

        buffer.write().unwrap().copy_from_slice(data);

        uploads.copy_buffer_to_image(vk::CopyBufferToImageInfo {
            regions: [vk::BufferImageCopy {
                buffer_row_length: rect.width(),
                buffer_image_height: rect.height(),
                image_offset: [rect.min.x, rect.min.y, 0],
                image_extent: [rect.width(), rect.height(), 1],
                image_subresource: cache_tex.subresource_layers(),
                ..Default::default()
            }].into(),
            ..vk::CopyBufferToImageInfo::buffer_image(buffer, cache_tex.clone())
        }).unwrap();
    }).unwrap();


    uploads.build().unwrap()
        .execute(app.transfer_queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    glyphs
}

use super::{MyVertex, WIDTH, HEIGHT};

pub fn render_glyphs(
    text: &String, pos: [f32; 2], size: f32, color: [f32; 4],
    vertices: &mut Vec<MyVertex>,
    fontr: &mut FontResources,
) {
    let mut vertices_to_append = 
        if let Some(existing_vertices) = fontr.text_render_cache.get(&**text) {
            existing_vertices.clone()
        } else {
            let new_render = vertices_for_text(&**text, fontr);
            fontr.text_render_cache.insert(text.clone(), new_render.clone());
            new_render
        };
    
    for vert in &mut vertices_to_append {
        vert.position[0] *= size;
        vert.position[0] += pos[0];
        vert.position[1] *= size;
        vert.position[1] += pos[1];
        vert.color = color;
    }

    vertices.extend(&vertices_to_append[..]);
}

fn vertices_for_text(text: &str, fontr: &mut FontResources) -> Vec<MyVertex> {
    let mut vertices = Vec::<MyVertex>::new();
    let mut pos: [f32; 2] = [0.,0.];

    let display_scale = 17.;

    pos[0] *= WIDTH as f32 / 2.;
    pos[1] *= HEIGHT as f32 / 2.;

    let mut last_glyph: Option<GlyphId> = None;
    let scale = Scale::uniform(1.);

    let font_v_metrics = Font::v_metrics(&fontr.font, scale);
    let height = (font_v_metrics.ascent + -font_v_metrics.descent) * display_scale;

    for char in text.chars() {
        let mut charindex = char as usize - 'a' as usize;
        if char == '(' { charindex = 26 }
        if char == ')' { charindex = 27 }
        let glyph = &fontr.glyphs[charindex];
        let rect = fontr.cache.rect_for(0, &glyph).expect("failed to get glyph");
        let placed_glyph = glyph.unpositioned().unscaled().clone()
            .scaled(scale);

        let glyph_id = glyph.id();

        let width = placed_glyph.h_metrics().advance_width * display_scale;

        if let Some((tex_rect, _)) = rect {
            // The official examples multiply these numbers by two. If we don't, the 
            // text ends up twice as small as it should. Not sure why.
            let bbox = placed_glyph.exact_bounding_box().unwrap();
            let screen_left =   (pos[0] + bbox.min.x * display_scale) / WIDTH  as f32 * 2.;
            let screen_right =  (pos[0] + bbox.max.x * display_scale) / WIDTH  as f32 * 2.;
            let screen_bottom = (pos[1] + bbox.max.y * display_scale) / HEIGHT as f32 * 2.;
            let screen_top =    (pos[1] + bbox.min.y * display_scale) / HEIGHT as f32 * 2.;
            //screen_bottom = placed_glyph

            let tex_left = tex_rect.min.x;
            let tex_right = tex_rect.max.x;
            let tex_top = tex_rect.min.y;
            let tex_bottom = tex_rect.max.y;

            let color: [f32; 4] = [0.,0.,0.,0.];

            let tl = MyVertex {
                position: [screen_left, screen_top],
                uv: [tex_left, tex_top],
                color,
            };
            let tr = MyVertex {
                position: [screen_right, screen_top],
                uv: [tex_right, tex_top],
                color,
            };
            let br = MyVertex {
                position: [screen_right, screen_bottom],
                uv: [tex_right, tex_bottom],
                color,
            };
            let bl = MyVertex {
                position: [screen_left, screen_bottom], 
                uv: [tex_left, tex_bottom], 
                color,
            };

            vertices.extend_from_slice(&[tl, tr, bl, bl, tr, br]);
        }

        if let Some(last) = last_glyph.take() {
            pos[0] += fontr.font.pair_kerning(scale, last, glyph_id) * display_scale;
        }
        pos[0] += placed_glyph.h_metrics().advance_width * display_scale;
    }

    return vertices;
}
