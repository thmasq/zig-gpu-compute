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

    // Kernel compilation to amdgcn
    const kernel_step = b.addSystemCommand(&.{
        "zig",     "build-obj",
        "-target", "amdgcn-amdhsa-none",
        "-mcpu=gfx1031", // Adjust for your GPU
        "--name",
        "kernel",
        "-O",
        "ReleaseFast",
        "-fno-strip",
        "src/kernel.zig",
    });

    b.installArtifact(exe);
    exe.step.dependOn(&kernel_step.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
