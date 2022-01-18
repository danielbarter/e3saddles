let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.jax
    (python3Packages.jaxlib.override { cudaSupport = true; })
    python3Packages.matplotlib
    nvtop
  ];

  LD_LIBRARY_PATH = with pkgs; builtins.concatStringsSep ":" [
    # cuda shared libraries
    "${cudatoolkit_11_2}/lib"
    "${cudatoolkit_11_2.lib}/lib"

    # nvidia driver shared libs
    "${pkgs.linuxPackages.nvidia_x11}/lib"
  ];
}
