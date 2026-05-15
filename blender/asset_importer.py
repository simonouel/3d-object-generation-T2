#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# --- Blender Add-on: Asset Importer ---
#
# Description:
# This add-on provides a UI in the 3D Viewport to import 3D models from a user-selected
# directory into Blender and add them to the Asset Browser. For each imported file
# (.fbx, .obj, .glb, .gltf):
# - Offsets the top-level object (typically an empty) by a cumulative 2 meters on the
#   Y-axis relative to the previous object (e.g., 0m, 2m, 4m, ...), adjusted by scale factor if enabled.
# - Identifies the first mesh child (geometry), renames it to match the filename
#   (without extension), marks it as an asset, and generates a preview.
# - Optionally applies a user-defined scale factor (0.1 to 10.0) to imported objects if enabled.
#
# Installation:
# 1. Save this script as `asset_importer.py`.
# 2. In Blender, go to Edit -> Preferences -> Add-ons -> Install, select the .py file,
#    and enable the add-on.
#
# Usage:
# 1. In the 3D Viewport, open the sidebar (N-key) and find the "Asset Importer" tab.
# 2. Click the folder icon to select the directory containing your asset files (defaults to %USERPROFILE%\.trellis\assets).
# 3. Check "Apply Model Scaling" to enable scaling, and set the scale factor (0.1 to 10.0).
# 4. Ensure you are in Object Mode.
# 5. Click "Import Assets" to run the import process.
# 6. Check the Blender System Console for progress and error messages
#    (Window -> Toggle System Console).
# 7. Verify assets and previews in the Asset Browser (Editor Type -> Asset Browser).
#
# Notes:
# - Requires Blender 3.0+ for asset marking and preview generation.
# - Only the first mesh child per file is renamed, marked as an asset, and given a preview.
# - Ensure the directory contains valid .fbx, .obj, .glb, or .gltf files.
# - Back up your Blender project before running.
# - Preview generation may take time; check the Asset Browser after completion.
# --------------------------------------------------------------------

import bpy
import os
from bpy.types import Operator, Panel
from bpy.props import StringProperty, BoolProperty, FloatProperty

# Add-on metadata
bl_info = {
    "name": "Asset Importer",
    "author": "NVIDIA",
    "version": (1, 2),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Asset Importer",
    "description": "Imports 3D models from a directory, offsets them, optionally scales them, and adds meshes to the Asset Browser",
    "category": "Import-Export",
}

# Operator to handle the import process
class ASSETIMPORTER_OT_import_assets(Operator):
    bl_idname = "assetimporter.import_assets"
    bl_label = "Import Assets"
    bl_description = "Import 3D models from the selected directory, offset them, optionally scale them, and add meshes to the Asset Browser"

    def execute(self, context):
        # Get the directory and scaling properties from the scene
        directory = context.scene.asset_importer_directory
        apply_scaling = context.scene.apply_model_scaling
        scale_factor = context.scene.model_scale_factor

        if not directory:
            self.report({'ERROR'}, "No directory selected. Please choose a directory.")
            return {'CANCELLED'}

        print(f"--- Starting Asset Import from: {directory} ---")
        if apply_scaling:
            print(f"Applying scale factor: {scale_factor}")

        if not os.path.isdir(directory):
            self.report({'ERROR'}, f"Directory not found: {directory}")
            print(f"Error: Directory not found: {directory}")
            return {'CANCELLED'}

        # Ensure we are in Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            print("Switched to Object Mode.")

        # Supported file extensions
        supported_extensions = ('.fbx', '.obj', '.glb', '.gltf')

        # Initialize cumulative Y-offset (base offset is 2.0 meters)
        base_offset = 2.0
        y_offset = 0.0

        imported_count = 0
        skipped_count = 0
        error_count = 0

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            if os.path.isfile(file_path) and file_ext in supported_extensions:
                print(f"\nProcessing file: {filename}")

                # Deselect all objects before importing
                bpy.ops.object.select_all(action='DESELECT')

                try:
                    # Import based on file extension
                    if file_ext == '.fbx':
                        bpy.ops.import_scene.fbx(filepath=file_path)
                    elif file_ext == '.obj':
                        bpy.ops.import_scene.obj(filepath=file_path)
                    elif file_ext in ('.glb', '.gltf'):
                        bpy.ops.import_scene.gltf(filepath=file_path)

                    # --- Post-Import Processing ---

                    # Get selected objects
                    imported_objects = context.selected_objects

                    if not imported_objects:
                        print(f"Warning: No objects selected after importing {filename}. Skipping rename/asset marking.")
                        error_count += 1
                        continue

                    # Assume first selected object is top-level (likely an empty)
                    top_level_obj = imported_objects[0]

                    # Apply scaling if enabled
                    if apply_scaling:
                        top_level_obj.scale = (scale_factor, scale_factor, scale_factor)
                        print(f"  Applied scale factor {scale_factor} to '{top_level_obj.name}'.")
                        # Apply the scale transform
                        bpy.ops.object.select_all(action='DESELECT')
                        top_level_obj.select_set(True)
                        context.view_layer.objects.active = top_level_obj
                        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                        print(f"  Applied scale transform for '{top_level_obj.name}'.")

                    # Apply cumulative Y-offset, adjusted by scale factor if scaling is enabled
                    effective_offset = y_offset * scale_factor if apply_scaling else y_offset
                    top_level_obj.location[1] = effective_offset
                    print(f"  Offset '{top_level_obj.name}' to {effective_offset}m on Y-axis.")

                    # Increment Y-offset (base offset adjusted by scale factor if enabled)
                    y_offset += base_offset

                    # Generate new name from filename (without extension)
                    base_name = os.path.splitext(filename)[0]

                    # Find mesh children
                    mesh_objects = [child for child in top_level_obj.children if child.type == 'MESH']

                    if not mesh_objects:
                        print(f"Warning: No mesh objects found under '{top_level_obj.name}' for {filename}. Skipping rename/asset marking.")
                        error_count += 1
                        continue

                    # Process the first mesh object
                    obj_to_process = mesh_objects[0]

                    # Deselect all for clean selection
                    bpy.ops.object.select_all(action='DESELECT')

                    # Select the mesh and make it active
                    obj_to_process.select_set(True)
                    context.view_layer.objects.active = obj_to_process

                    # Rename the mesh
                    old_name = obj_to_process.name
                    obj_to_process.name = base_name
                    print(f"  Renamed '{old_name}' to '{base_name}'")

                    # Mark as asset and generate preview
                    try:
                        obj_to_process.asset_mark()
                        print(f"  Marked '{base_name}' as an asset.")
                        obj_to_process.asset_generate_preview()
                        print(f"  Generated preview for '{base_name}'.")
                    except Exception as e:
                        print(f"  Error marking or generating preview for '{base_name}': {e}")
                        error_count += 1

                    imported_count += 1

                except Exception as e:
                    print(f"Error importing or processing file {filename}: {e}")
                    error_count += 1
                    bpy.ops.object.select_all(action='DESELECT')

            elif os.path.isfile(file_path):
                # print(f"Skipping unsupported file type: {filename}")
                skipped_count += 1

        print(f"\n--- Import Process Finished ---")
        print(f"Successfully imported and marked: {imported_count}")
        print(f"Skipped (unsupported type): {skipped_count}")
        print(f"Errors encountered: {error_count}")

        # Show summary in UI
        self.report({'INFO'}, f"Imported: {imported_count}, Skipped: {skipped_count}, Errors: {error_count}")
        bpy.context.window_manager.popup_menu(
            lambda self, ctx: self.layout.label(text=f"Import Complete! Imported: {imported_count}, Skipped: {skipped_count}, Errors: {error_count}"),
            title="Import Summary", icon='INFO'
        )

        return {'FINISHED'}

# UI Panel for the add-on
class ASSETIMPORTER_PT_panel(Panel):
    bl_label = "Asset Importer"
    bl_idname = "PT_AssetImporter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Asset Importer"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Directory picker
        layout.prop(scene, "asset_importer_directory", text="Directory")

        # --- New UI Elements ---
        # Checkbox for applying model scaling
        layout.prop(scene, "apply_model_scaling", text="Apply Model Scaling")

        # Scale factor spinner, enabled only if apply_model_scaling is checked
        row = layout.row()
        row.enabled = scene.apply_model_scaling
        row.prop(scene, "model_scale_factor", text="Scale Factor")

        # Import button
        layout.operator("assetimporter.import_assets", text="Import Assets")

# Register classes and properties
def register():
    bpy.utils.register_class(ASSETIMPORTER_OT_import_assets)
    bpy.utils.register_class(ASSETIMPORTER_PT_panel)
    
    # Add directory property to Scene with default %USERPROFILE%\.trellis\assets
    bpy.types.Scene.asset_importer_directory = StringProperty(
        name="Import Directory",
        description="Directory containing 3D model files to import",
        subtype='DIR_PATH',
        default=os.path.expandvars(r"%USERPROFILE%\.trellis\assets")
    )

    # --- New Properties ---
    # Checkbox for enabling/disabling model scaling
    bpy.types.Scene.apply_model_scaling = BoolProperty(
        name="Apply Model Scaling",
        description="Apply a uniform scale factor to imported objects",
        default=False
    )

    # Scale factor property (0.1 to 10.0)
    bpy.types.Scene.model_scale_factor = FloatProperty(
        name="Scale Factor",
        description="Scale factor to apply to imported objects (0.1 to 10.0)",
        default=1.0,
        min=0.1,
        max=100.0,
        step=100  # 0.1 increments in the UI
    )

def unregister():
    bpy.utils.unregister_class(ASSETIMPORTER_OT_import_assets)
    bpy.utils.unregister_class(ASSETIMPORTER_PT_panel)
    
    # Remove properties
    del bpy.types.Scene.asset_importer_directory
    del bpy.types.Scene.apply_model_scaling
    del bpy.types.Scene.model_scale_factor

if __name__ == "__main__":
    register()