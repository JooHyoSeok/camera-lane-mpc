"""Make MetaDrive 0.4.3 render correctly on Apple Silicon macOS.
Import this module before creating an env.

Problems on macOS OpenGL, and the fixes applied here (no files are edited):
  1. No pixel format exists for MetaDrive's multisampling defaults (8x MSAA
     window, 16x MSAA float16 buffers in the PBR pipeline and the RGB camera),
     so the window and the tone-map passes fail to open. -> clamp to 4x.
  2. The default context is legacy OpenGL 2.1 (GLSL 1.20), but the road/terrain
     shader is GLSL 3.30, so roads and lane markings render blank. -> request a
     3.2 core context.
  3. Under a core context, GLSL 1.20 shaders no longer compile: the PBR/tone-map
     shaders (simplepbr), the Mac sky box, and texture2D() calls inside the
     terrain shader. -> use simplepbr's 330 variant, the non-Mac sky box, and
     rewrite texture2D() -> texture() when loading shaders.

MetaDrive also cannot render offscreen on macOS: any camera sensor or
use_render=True opens a real window.
"""
import re

from direct.filter.FilterManager import FilterManager
from panda3d.core import Shader, loadPrcFileData

import metadrive.component.sensors.rgb_camera as _rgb_camera
import metadrive.engine.core.engine_core as _engine_core  # sets "multisamples 8" at import
import metadrive.engine.core.sky_box as _sky_box
import metadrive.engine.core.terrain as _terrain
import metadrive.third_party.simplepbr as _simplepbr

MAX_SAMPLES = 4
loadPrcFileData("", f"multisamples {MAX_SAMPLES}")  # later pages override earlier ones
loadPrcFileData("", "gl-version 3 2")

# 1) multisampling
_render_scene_into = FilterManager.render_scene_into


def _render_scene_into_mac(self, *args, fbprops=None, **kwargs):
    if fbprops is not None and fbprops.get_multisamples() > MAX_SAMPLES:
        fbprops.set_multisamples(MAX_SAMPLES)
    return _render_scene_into(self, *args, fbprops=fbprops, **kwargs)


FilterManager.render_scene_into = _render_scene_into_mac


# 3) GLSL for a core-profile context
def _core_glsl(src):
    src = re.sub(r"\btexture2D\s*\(", "texture(", src)
    src = re.sub(r"\btextureCube\s*\(", "texture(", src)
    return src


_load_shader_str = _simplepbr._load_shader_str


def _load_shader_str_330(shaderpath, defines=None):
    defines = dict(defines or {})
    defines["USE_330"] = ""
    return _core_glsl(_load_shader_str(shaderpath, defines))


_simplepbr._load_shader_str = _load_shader_str_330
_rgb_camera._load_shader_str = _load_shader_str_330

_pbr_init = _engine_core.init


def _pbr_init_mac(**kwargs):
    kwargs["use_330"] = True
    return _pbr_init(**kwargs)


_engine_core.init = _pbr_init_mac


class _CoreShader:
    """Stand-in for panda3d's Shader inside MetaDrive modules: same API, but
    file-based GLSL is patched for a core context before compiling."""
    SL_GLSL = Shader.SL_GLSL

    @staticmethod
    def load(lang, vertex="", fragment="", *rest, **kw):
        if lang == Shader.SL_GLSL and not rest and not kw:
            def read(p):
                with open(str(p)) as f:
                    return _core_glsl(f.read())
            return Shader.make(Shader.SL_GLSL, vertex=read(vertex), fragment=read(fragment))
        return Shader.load(lang, vertex, fragment, *rest, **kw)

    make = staticmethod(Shader.make)


_terrain.Shader = _CoreShader
_sky_box.Shader = _CoreShader
_sky_box.is_mac = lambda: False  # the "_mac" sky box is GLSL 1.20; use the 1.50 one
