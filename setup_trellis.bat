@echo off

REM Setup trellis submodule excluding INRIA-licensed files and applying patches

REM Check if trellis submodule is already properly set up
if exist "trellis\.git" (
    REM Check if .git is a file (absorbed submodule) - means setup was already done
    for %%A in ("trellis\.git") do (
        if not "%%~aA"=="d---------" (
            echo Trellis submodule already set up, applying patches...
            goto :apply_patches
        )
    )
)

REM Remove existing trellis directory if it exists but isn't properly set up
if exist "trellis" (
    echo Removing incomplete trellis directory...
    rmdir /s /q trellis >nul 2>&1
)

git submodule init trellis >nul 2>&1

for /f "tokens=*" %%i in ('git config --get submodule.trellis.url') do set "SUBMODULE_URL=%%i"
git clone --no-checkout %SUBMODULE_URL% trellis >nul 2>&1

cd trellis
git config core.sparseCheckout true
if not exist ".git\info" mkdir ".git\info"
echo /*> ".git\info\sparse-checkout"
echo ^!trellis/representations/gaussian/general_utils.py>> ".git\info\sparse-checkout"
echo ^!trellis/representations/gaussian/gaussian_model.py>> ".git\info\sparse-checkout"
echo ^!trellis/renderers/gaussian_render.py>> ".git\info\sparse-checkout"
echo ^!trellis/renderers/mesh_renderer.py>> ".git\info\sparse-checkout"
echo ^!trellis/representations/mesh/flexicubes/>> ".git\info\sparse-checkout"
echo ^!trellis/representations/mesh/cube2mesh.py>> ".git\info\sparse-checkout"
echo ^!trellis/pipelines/samplers/flow_euler.py>> ".git\info\sparse-checkout"
echo ^!trellis/utils/render_utils.py>> ".git\info\sparse-checkout"
echo ^!trellis/utils/postprocessing_utils.py>> ".git\info\sparse-checkout"

cd ..
for /f "tokens=3" %%i in ('git ls-tree HEAD trellis') do set "EXPECTED_COMMIT=%%i"
cd trellis
git checkout %EXPECTED_COMMIT% >nul 2>&1
cd ..

git submodule absorbgitdirs trellis >nul 2>&1

:apply_patches

REM Note: We skip --recursive because FlexiCubes is the only nested submodule
REM and we're using our own patched version instead

REM Copy MIT/Apache-2.0 licensed patch files (removes easydict and kaolin dependencies)
copy /Y "trellis_patch\renderers\gaussian_render.py" "trellis\trellis\renderers\" >nul 2>&1
copy /Y "trellis_patch\renderers\mesh_renderer.py" "trellis\trellis\renderers\" >nul 2>&1
copy /Y "trellis_patch\representations\gaussian\gaussian_model.py" "trellis\trellis\representations\gaussian\" >nul 2>&1
copy /Y "trellis_patch\representations\mesh\cube2mesh.py" "trellis\trellis\representations\mesh\" >nul 2>&1
copy /Y "trellis_patch\pipelines\samplers\flow_euler.py" "trellis\trellis\pipelines\samplers\" >nul 2>&1
copy /Y "trellis_patch\utils\render_utils.py" "trellis\trellis\utils\" >nul 2>&1
copy /Y "trellis_patch\utils\postprocessing_utils.py" "trellis\trellis\utils\" >nul 2>&1

REM Create flexicubes directory and copy patched files (replaces FlexiCubes submodule entirely)
if not exist "trellis\trellis\representations\mesh\flexicubes" mkdir "trellis\trellis\representations\mesh\flexicubes"
copy /Y "trellis_patch\representations\mesh\flexicubes\flexicubes.py" "trellis\trellis\representations\mesh\flexicubes\" >nul 2>&1
copy /Y "trellis_patch\representations\mesh\flexicubes\tables.py" "trellis\trellis\representations\mesh\flexicubes\" >nul 2>&1

exit /b 0
