// ============================================
// SHADER DE RENDU (Render Pipeline)
// ============================================
// Ce shader dessine le tissu à l'écran
// - Vertex shader : transforme les positions 3D en positions écran
// - Fragment shader : colore les pixels

// ============================================
// UNIFORMS (Caméra)
// ============================================
struct Camera {
    mvp: mat4x4<f32>,  // Model-View-Projection matrix
};

@group(0) @binding(0) var<uniform> cam: Camera;

// ============================================
// VERTEX SHADER INPUT
// ============================================
struct VSIn {
    @location(0) position: vec4<f32>,  // Position du vertex (x, y, z, w)
};

// ============================================
// VERTEX SHADER OUTPUT
// ============================================
struct VSOut {
    @builtin(position) clip: vec4<f32>,  // Position écran (clip space)
    @location(0) world_pos: vec3<f32>,   // Position monde (pour coloration)
};

// ============================================
// VERTEX SHADER
// ============================================
@vertex
fn vs_main(in: VSIn) -> VSOut {
    var out: VSOut;
    
    // Transformation Model-View-Projection
    out.clip = cam.mvp * in.position;
    
    // Passer la position monde au fragment shader
    out.world_pos = in.position.xyz;
    
    return out;
}

// ============================================
// FRAGMENT SHADER
// ============================================
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Coloration simple basée sur la hauteur (Y)
    // Plus haut = plus clair
    let height_factor = (in.world_pos.y + 2.0) * 0.3;
    let base_color = vec3<f32>(0.9, 0.9, 0.9);
    let color = base_color * (0.5 + height_factor);
    
    return vec4<f32>(color, 1.0);
}