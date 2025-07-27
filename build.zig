const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Host executable
    const exe = b.addExecutable(.{
        .name = "gpu_compute",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link HSA runtime - Arch Linux specific paths
    exe.addLibraryPath(.{ .cwd_relative = "/opt/rocm/lib" });
    exe.addIncludePath(.{ .cwd_relative = "/opt/rocm/include" });
    exe.linkSystemLibrary("hsa-runtime64");
    exe.linkLibC();

    b.installArtifact(exe);

    // Step 1: Generate LLVM IR from Zig kernel
    const ir_step = b.addSystemCommand(&.{
        "zig",                      "build-obj",
        "-target",                  "amdgcn-amdhsa-none",
        "-mcpu=gfx1031",            "-O",
        "ReleaseFast",              "-fno-strip",
        "-femit-llvm-ir=kernel.ll", "-fno-emit-bin",
        "src/kernel.zig",
    });

    const hsa_step = b.addSystemCommand(&.{
        "/usr/lib/llvm19/bin/clang", "-x",                      "ir",
        "-target",                   "amdgcn-amd-amdhsa",       "-mcpu=gfx1031",
        "-O3",                       "-mcode-object-version=4", "-o",
        "kernel.o",                  "kernel.ll",
    });
    hsa_step.step.dependOn(&ir_step.step);

    exe.step.dependOn(&hsa_step.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
