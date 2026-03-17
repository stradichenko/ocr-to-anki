# NixOS runtime support for the pre-built OCR-to-Anki release binary.
#
# The CI builds the Flutter app on Ubuntu, which links against system
# libraries (GTK 3, glib, …) in standard FHS paths (/usr/lib).  NixOS
# keeps these libraries in the Nix store, so the binary cannot find them.
#
# This file provides a nix-shell environment that:
#   1. Exports LD_LIBRARY_PATH with every needed runtime library.
#   2. Patches the ELF interpreter to the NixOS dynamic linker.
#
# Usage (automatic):  ./run.sh               # detects NixOS
# Usage (manual):     nix-shell nixos-support.nix --run ./ocr_to_anki

{ pkgs ? import <nixpkgs> {} }:

let
  runtimeLibs = with pkgs; [
    # GTK 3 toolkit (Flutter Linux embedder)
    gtk3
    glib
    pango
    harfbuzz
    cairo
    gdk-pixbuf
    atk
    libepoxy

    # X11
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
    xorg.libXext
    xorg.libXfixes
    xorg.libXinerama
    xorg.libXdamage
    xorg.libXcomposite
    xorg.libXrender

    # Wayland
    wayland
    libxkbcommon

    # GL / EGL
    libGL
    mesa

    # System
    zlib
    pcre2
    stdenv.cc.cc.lib        # libstdc++
    dbus.lib
    fontconfig.lib
    freetype
  ];
in

pkgs.mkShell {
  nativeBuildInputs = [ pkgs.patchelf ];
  buildInputs = runtimeLibs;

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # Patch the ELF interpreter so the Ubuntu-built binary runs on NixOS
    if [ -n "''${OCR_TO_ANKI_BIN:-}" ] && [ -f "''${OCR_TO_ANKI_BIN}" ]; then
      _DESIRED_INTERP=$(cat $NIX_CC/nix-support/dynamic-linker)
      _CURRENT_INTERP=$(patchelf --print-interpreter "''${OCR_TO_ANKI_BIN}" 2>/dev/null || echo "")
      if [ "$_CURRENT_INTERP" != "$_DESIRED_INTERP" ]; then
        patchelf --set-interpreter "$_DESIRED_INTERP" "''${OCR_TO_ANKI_BIN}" 2>/dev/null || true
        echo "Patched ELF interpreter → $_DESIRED_INTERP"
      fi
    fi
  '';
}
