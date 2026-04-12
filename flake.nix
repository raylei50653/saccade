{
  description = "YOLO-LLM Development Environment (CUDA + GStreamer)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python & uv
            python311
            uv

            # CUDA
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
            cudaPackages.tensorrt

            # GStreamer
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-bad
            gst_all_1.gst-plugins-ugly
            gst_all_1.gst-vaapi
          ];

          shellHook = ''
            echo "YOLO-LLM environment loaded with CUDA and GStreamer"
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.tensorrt}/lib
          '';
        };
      }
    );
}
