@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> partial_sums: array<u32>;    
var<workgroup> shared_data: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>) {

    let idx = global_id.x * 2u; // Each thread processes 2 elements
    let local_index = local_id.x;

    // Load data into shared memory
    shared_data[local_index] = input[idx] + input[idx + 1u];
    workgroupBarrier();

    // Reduction loop
    var stride = 1u;
    while (stride < 64u) {
        let index = 2u * stride * local_index;
        if (index + stride < 64u) {
            shared_data[index] += shared_data[index + stride];
        }
        workgroupBarrier();
        stride = stride * 2u;
    }

    // Write result of this workgroup
    if (local_index == 0u) {
        partial_sums[group_id.x] = shared_data[0];
    }
}