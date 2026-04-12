{
  description = "Saccade Development Environment (CUDA + GStreamer + UV)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    nix-direnv.url = "github:nix-community/nix-direnv";
  };

  outputs = { self, nixpkgs, flake-utils, nix-direnv }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = (system == "x86_64-linux");
        };
      in
      {
        # 完整 GPU 開發環境 (本機使用)
        devShells.default = pkgs.mkShell {
          name = "saccade-gpu-shell";
          buildInputs = with pkgs; [
            # Python & uv
            python312
            uv

            # CUDA & AI
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
            cudaPackages.cuda_nvcc
            llama-cpp
            
            # GStreamer & Media
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-bad
            gst_all_1.gst-plugins-ugly
            gst_all_1.gst-vaapi
            ffmpeg_6-full

            # System Libraries (Required for OpenCV, CUDA, and LLM backends)
            zlib
            glib
            libGL
            stdenv.cc.cc.lib
            linuxPackages.nvidia_x11
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.glib
              pkgs.libGL
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.linuxPackages.nvidia_x11
            ]}"
            
            # CUDA configuration
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I${pkgs.cudaPackages.cudatoolkit}/include"

            # Python path for local modules
            export PYTHONPATH=$PYTHONPATH:$(pwd)

            echo "🚀 Saccade GPU environment loaded (CUDA + GStreamer + Python 3.12)"
            echo "💡 Tip: Run 'uv sync' to initialize Python dependencies."
            echo "⚠️  MediaMTX not managed by Nix — see infra/mediamtx.yml"
          '';
        };

        # 輕量級 CI 環境 (GitHub Actions 使用，排除 GPU 驅動)
        devShells.ci = pkgs.mkShell {
          name = "saccade-ci-shell";
          buildInputs = with pkgs; [
            python312
            uv
            zlib
            glib
          ];
          shellHook = ''
            export PYTHONPATH=$PYTHONPATH:$(pwd)
            echo "✅ Saccade CI environment loaded (No GPU)"
          '';
        };
      }
    );
}
