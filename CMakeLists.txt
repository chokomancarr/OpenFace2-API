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
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CMAKE_CURRENT_SOURCE_DIR}/openface2/LandmarkDetector/model/"
	${EXE_PATH}/model/
)

# Move OpenCV classifiers
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CMAKE_CURRENT_SOURCE_DIR}/lib/3rdParty/OpenCV/classifiers/"
	${EXE_PATH}/classifiers/
)

# Move AU prediction modules
add_custom_command(TARGET copy_datasets POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CMAKE_CURRENT_SOURCE_DIR}/openface2/FaceAnalyser/AU_predictors/"
	${EXE_PATH}/AU_predictors/
)

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
