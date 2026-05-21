import os
import sys
import datetime
import faulthandler
import ctypes
import shutil


def _log_path() -> str:
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.getcwd()
    log_dir = os.path.join(base, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(log_dir, "runtime.log")


def _find_qt_plugins(base: str) -> str:
    candidates = [
        os.path.join(base, "PyQt5", "Qt5", "plugins"),
        os.path.join(base, "PyQt5", "Qt", "plugins"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def _find_qt_bin(base: str) -> str:
    candidates = [
        os.path.join(base, "PyQt5", "Qt5", "bin"),
        os.path.join(base, "PyQt5", "Qt", "bin"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def _to_short_path(path: str) -> str:
    try:
        if not path or not isinstance(path, str):
            return path
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
        if n and n < len(buf):
            return buf.value
    except Exception:
        pass
    return path


def _prepare_min_plugins(src_plugins: str, dst_root: str) -> str:
    if not src_plugins or not os.path.isdir(src_plugins):
        return ""
    if not dst_root:
        return ""
    dst_plugins = os.path.join(dst_root, "qt_plugins_min")
    marker = os.path.join(dst_plugins, ".ready")
    try:
        if os.path.isfile(marker):
            return dst_plugins
    except Exception:
        pass
    try:
        os.makedirs(dst_plugins, exist_ok=True)
        needed = ["platforms", "imageformats", "iconengines", "styles"]
        for name in needed:
            src_dir = os.path.join(src_plugins, name)
            dst_dir = os.path.join(dst_plugins, name)
            if os.path.isdir(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        # Keep only qwindows platform plugin to avoid extra loaders.
        try:
            plat_dir = os.path.join(dst_plugins, "platforms")
            if os.path.isdir(plat_dir):
                for fn in os.listdir(plat_dir):
                    low = fn.lower()
                    if not (low == "qwindows.dll" or low == "qwindowsd.dll"):
                        try:
                            os.remove(os.path.join(plat_dir, fn))
                        except Exception:
                            pass
        except Exception:
            pass
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok")
        return dst_plugins
    except Exception:
        return dst_plugins if os.path.isdir(dst_plugins) else ""


def _prune_qt_plugins(base: str) -> None:
    try:
        if not base:
            return
        for rel in [
            os.path.join("PyQt5", "Qt5", "plugins", "platformthemes"),
            os.path.join("PyQt5", "Qt5", "plugins", "accessible"),
            os.path.join("PyQt5", "Qt", "plugins", "platformthemes"),
            os.path.join("PyQt5", "Qt", "plugins", "accessible"),
        ]:
            target = os.path.join(base, rel)
            try:
                shutil.rmtree(target, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def _remove_qt_conf(base: str) -> None:
    try:
        if not base:
            return
        for rel in [
            os.path.join("PyQt5", "Qt5", "qt.conf"),
            os.path.join("PyQt5", "Qt", "qt.conf"),
        ]:
            p = os.path.join(base, rel)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def _setup_qt_env():
    if not getattr(sys, "frozen", False):
        return "", ""
    base = getattr(sys, "_MEIPASS", None)
    if not isinstance(base, str) or not base:
        return "", ""
    # Prefer short path to avoid unicode path edge cases
    base = _to_short_path(base)
    _prune_qt_plugins(base)
    _remove_qt_conf(base)
    qt_plugins_src = _find_qt_plugins(base)
    qt_bin = _find_qt_bin(base)
    qt_plugins_src = _to_short_path(qt_plugins_src)
    qt_bin = _to_short_path(qt_bin)
    exe_dir = _to_short_path(os.path.dirname(sys.executable))
    qt_plugins = _prepare_min_plugins(qt_plugins_src, exe_dir)
    qt_plugins = _to_short_path(qt_plugins)
    if qt_plugins:
        # Force Qt to use minimal plugins only (avoid platformthemes/accessible).
        os.environ["QT_PLUGIN_PATH"] = qt_plugins
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plugins, "platforms")
        os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    if qt_bin:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(qt_bin)
        except Exception:
            pass
        os.environ["PATH"] = qt_bin + os.pathsep + os.environ.get("PATH", "")
    # Reduce accessibility and OpenGL edge cases
    os.environ.setdefault("QT_ACCESSIBILITY", "0")
    os.environ.setdefault("QT_ENABLE_ACCESSIBILITY", "0")
    os.environ.setdefault("QT_NO_ACCESSIBILITY", "1")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QT_QPA_PLATFORMTHEME", "windows")
    os.environ.setdefault("QT_STYLE_OVERRIDE", "Fusion")
    # Debug plugins only when not explicitly disabled
    os.environ.setdefault("QT_DEBUG_PLUGINS", "1")
    return qt_plugins, qt_bin


def _setup_extra_dll_dirs():
    """Add pandas/numpy libs to DLL search path (frozen only)."""
    if not getattr(sys, "frozen", False):
        return []
    base = getattr(sys, "_MEIPASS", None)
    if not isinstance(base, str) or not base:
        return []
    base = _to_short_path(base)
    exe_dir = _to_short_path(os.path.dirname(sys.executable))
    candidates = [
        os.path.join(base, "pandas.libs"),
        os.path.join(base, "numpy.libs"),
        os.path.join(exe_dir, "pandas.libs"),
        os.path.join(exe_dir, "numpy.libs"),
    ]
    added = []
    for p in candidates:
        if not p or not os.path.isdir(p):
            continue
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(p)
        except Exception:
            pass
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
        added.append(p)
    return added


_qt_plugins, _qt_bin = _setup_qt_env()
_extra_dll_dirs = _setup_extra_dll_dirs()

# Use short path for CWD when frozen
try:
    if getattr(sys, "frozen", False):
        exe_dir = _to_short_path(os.path.dirname(sys.executable))
        if exe_dir:
            os.chdir(exe_dir)
        # Write qt.conf to force plugin path to minimal bundle
        try:
            if exe_dir and _qt_plugins:
                qt_conf = os.path.join(exe_dir, "qt.conf")
                with open(qt_conf, "w", encoding="utf-8") as f:
                    f.write("[Paths]\n")
                    f.write("Prefix=.\n")
                    f.write("Plugins=qt_plugins_min\n")
        except Exception:
            pass
except Exception:
    pass

_path = _log_path()
try:
    _fp = open(_path, "a", encoding="utf-8", buffering=1)
    _fp.write("\n=== App start: %s ===\n" % datetime.datetime.now().isoformat())
    sys.stdout = _fp
    sys.stderr = _fp
    faulthandler.enable(file=_fp)
    _fp.write("[runtime] QT_PLUGIN_PATH=%s\n" % os.environ.get("QT_PLUGIN_PATH"))
    _fp.write("[runtime] QT_QPA_PLATFORM_PLUGIN_PATH=%s\n" % os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"))
    _fp.write("[runtime] QT_QPA_PLATFORM=%s\n" % os.environ.get("QT_QPA_PLATFORM"))
    _fp.write("[runtime] QT_PLUGINS_RESOLVED=%s\n" % _qt_plugins)
    _fp.write("[runtime] QT_BIN_RESOLVED=%s\n" % _qt_bin)
    _fp.write("[runtime] QT_OPENGL=%s\n" % os.environ.get("QT_OPENGL"))
    _fp.write("[runtime] QT_ACCESSIBILITY=%s\n" % os.environ.get("QT_ACCESSIBILITY"))
    _fp.write("[runtime] QT_NO_ACCESSIBILITY=%s\n" % os.environ.get("QT_NO_ACCESSIBILITY"))
    _fp.write("[runtime] QT_QPA_PLATFORMTHEME=%s\n" % os.environ.get("QT_QPA_PLATFORMTHEME"))
    _fp.write("[runtime] QT_STYLE_OVERRIDE=%s\n" % os.environ.get("QT_STYLE_OVERRIDE"))
    _fp.write("[runtime] QT_PLUGIN_PATH_USED=%s\n" % os.environ.get("QT_PLUGIN_PATH"))
    _fp.write("[runtime] EXTRA_DLL_DIRS=%s\n" % _extra_dll_dirs)
    # Try to read Qt library paths if available
    try:
        from PyQt5 import QtCore  # noqa: F401
        try:
            if _qt_plugins:
                QtCore.QCoreApplication.setLibraryPaths([_qt_plugins])
            def _qt_message_handler(mode, context, message):
                try:
                    _fp.write("[Qt] %s\n" % message)
                    _fp.flush()
                except Exception:
                    pass

            QtCore.qInstallMessageHandler(_qt_message_handler)
        except Exception:
            pass
        try:
            paths = QtCore.QCoreApplication.libraryPaths()
            _fp.write("[runtime] Qt libraryPaths=%s\n" % paths)
        except Exception:
            pass
    except Exception:
        pass
except Exception:
    pass
