cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project(OpenFace VERSION 2.0.2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

set(CMAKE_CONFIG_DIR etc/OpenFace)
set(CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_CONFIG_DIR}")
add_definitions(-DCONFIG_DIR="${CONFIG_DIR}")

# make sure we'll use OpenBLAS only: there's a header file naming difference between different
# implementations; so OpenFace wants OpenBLAS;
find_package( OpenBLAS REQUIRED)
find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect)
find_package( Boost 1.5.9 COMPONENTS filesystem system)

add_custom_target(copy_datasets)
if (NOT ${EXE_PATH})
	set(${EXE_PATH} ${CMAKE_BINARY_DIR}/bin/)
endif()

# Move LandmarkDetector model
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory  ${EXE_PATH}/model/)
file(GLOB files "openface2/LandmarkDetector/model/*.txt")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/)

# Move the hierarchical LandmarkDetector models
file(GLOB files "openface2/LandmarkDetector/model/model*")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/)

# Move detection validation models
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory  ${EXE_PATH}/model/detection_validation/)
file(GLOB files "openface2/LandmarkDetector/model/detection_validation/*.txt")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/detection_validation/)

# Move patch experts
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory  ${EXE_PATH}/model/patch_experts/)
file(GLOB files "openface2/LandmarkDetector/model/patch_experts/*.txt")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/patch_experts/)

# Move CEN patch experts
file(GLOB files "openface2/LandmarkDetector/model/patch_experts/*.dat")
if ("${files}" STREQUAL "")
	message(FATAL_ERROR "missing files in LandmarkDetector/model/patch_experts/! you may need to run the download_models script!")
else()
	add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/patch_experts/)
endif()

# Move MTCNN face detector
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory  ${EXE_PATH}/model/mtcnn_detector/)
file(GLOB files "openface2/LandmarkDetector/model/mtcnn_detector/*.txt")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/mtcnn_detector/)

# Move MTCNN face detector
file(GLOB files "openface2/LandmarkDetector/model/mtcnn_detector/*.dat")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/mtcnn_detector/)

# Move Point Distribution models
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory  ${EXE_PATH}/model/pdms/)
file(GLOB files "openface2/LandmarkDetector/model/pdms/*.txt")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/model/pdms/)

# Move OpenCV classifiers
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory  ${EXE_PATH}/classifiers/)
file(GLOB files "lib/3rdParty/OpenCV/classifiers/*.xml")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/classifiers/)

# Move AU prediction modules
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory  ${EXE_PATH}/AU_predictors/)
file(GLOB files "openface2/FaceAnalyser/AU_predictors/*.txt")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/AU_predictors/)

# Move AU prediction modules
file(GLOB files "openface2/FaceAnalyser/AU_predictors/svr*")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/AU_predictors/)

# Move AU prediction modules
file(GLOB files "openface2/FaceAnalyser/AU_predictors/svm*")
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} ${EXE_PATH}/AU_predictors/)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 8.0)
		MESSAGE(FATAL_ERROR "Need a 8.0 or newer GCC compiler. Current GCC: ${GCC_VERSION}")
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")
    endif ()
endif ()

#deleted: use submodule dlib because installed dlib is a pain

# suppress auto_ptr deprecation warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options("-Wno-deprecated-declarations")
endif()

# LandmarkDetector library
add_subdirectory(openface2/LandmarkDetector)
# Facial Expression analysis library
add_subdirectory(openface2/FaceAnalyser)
# Gaze estimation library
add_subdirectory(openface2/GazeAnalyser)
# Utilities library
add_subdirectory(openface2/Utilities)
