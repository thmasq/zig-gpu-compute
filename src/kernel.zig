const std = @import("std");
const builtin = @import("builtin");

// AMDGCN intrinsics as external functions
extern fn @"llvm.amdgcn.workitem.id.x"() u32;
extern fn @"llvm.amdgcn.workitem.id.y"() u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() u32;
extern fn @"llvm.amdgcn.workgroup.id.y"() u32;
extern fn @"llvm.amdgcn.s.barrier"() void;

// Shared memory declaration using addrspace
var shared_mem: [512]f32 addrspace(.shared) = undefined;

// Helper functions for safe shared memory access
inline fn setSharedMem(index: u32, value: f32) void {
    if (index < 512) {
        shared_mem[index] = value;
    }
}

inline fn getSharedMem(index: u32) f32 {
    if (index < 512) {
        return shared_mem[index];
    }
    return 0.0;
}

// Kernel attributes for AMD GPU
export fn vector_add_shared(a: [*]addrspace(.global) const f32, b: [*]addrspace(.global) const f32, c: [*]addrspace(.global) f32, n: u32) callconv(.Kernel) void {
    // Correct ID usage
    const local_id = @"llvm.amdgcn.workitem.id.x"(); // 0-255 within workgroup
    const group_id = @"llvm.amdgcn.workgroup.id.x"(); // Workgroup index

    const lsize: u32 = 256; // Workgroup size
    const global_id = group_id * lsize + local_id;

    if (global_id >= n) return;

    // Load data into shared memory
    shared_mem[local_id] = a[global_id] + b[global_id];

    // Synchronize workgroup
    @"llvm.amdgcn.s.barrier"();

    // Perform reduction in shared memory
    var stride: u32 = lsize / 2;
    while (stride > 0) : (stride /= 2) {
        if (local_id < stride and local_id + stride < lsize) {
            shared_mem[local_id] += shared_mem[local_id + stride];
        }
        @"llvm.amdgcn.s.barrier"();
    }

    // Only thread 0 writes the reduced result for this workgroup
    if (local_id == 0) {
        c[group_id] = shared_mem[0]; // One result per workgroup
    }
}

export fn matrix_multiply_shared(a: [*]addrspace(.global) const f32, b: [*]addrspace(.global) const f32, c: [*]addrspace(.global) f32, width: u32) callconv(.Kernel) void {
    const local_x = @"llvm.amdgcn.workitem.id.x"();
    const local_y = @"llvm.amdgcn.workitem.id.y"();
    const group_x = @"llvm.amdgcn.workgroup.id.x"();
    const group_y = @"llvm.amdgcn.workgroup.id.y"();

    const TILE_SIZE = 16;

    // Bounds check early
    const row = group_y * TILE_SIZE + local_y;
    const col = group_x * TILE_SIZE + local_x;

    if (row >= width or col >= width) return;

    const a_tile_base: u32 = 0;
    const b_tile_base: u32 = 256;

    var c_val: f32 = 0.0;

    // Tile across the K dimension
    var tile: u32 = 0;
    while (tile < (width + TILE_SIZE - 1) / TILE_SIZE) : (tile += 1) {
        // Load tiles into shared memory with bounds checking
        const a_col = tile * TILE_SIZE + local_x;
        const b_row = tile * TILE_SIZE + local_y;

        // Load A tile element
        const a_tile_idx = a_tile_base + local_y * TILE_SIZE + local_x;
        if (row < width and a_col < width) {
            setSharedMem(a_tile_idx, a[row * width + a_col]);
        } else {
            setSharedMem(a_tile_idx, 0.0);
        }

        // Load B tile element
        const b_tile_idx = b_tile_base + local_y * TILE_SIZE + local_x;
        if (b_row < width and col < width) {
            setSharedMem(b_tile_idx, b[b_row * width + col]);
        } else {
            setSharedMem(b_tile_idx, 0.0);
        }

        @"llvm.amdgcn.s.barrier"();

        // Compute partial result using linear indexing
        var k: u32 = 0;
        while (k < TILE_SIZE) : (k += 1) {
            const a_val = getSharedMem(a_tile_base + local_y * TILE_SIZE + k);
            const b_val = getSharedMem(b_tile_base + k * TILE_SIZE + local_x);
            c_val += a_val * b_val;
        }

        @"llvm.amdgcn.s.barrier"();
    }

    // Write result with bounds check
    if (row < width and col < width) {
        c[row * width + col] = c_val;
    }
}
