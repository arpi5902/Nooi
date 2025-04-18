{pkgs}: {
  deps = [
    pkgs.rapidfuzz-cpp
    pkgs.python312Packages.pyspellchecker
    pkgs.glibcLocales
    pkgs.libyaml
    pkgs.ffmpeg
    pkgs.wget
  ];
}
