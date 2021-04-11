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

# Move LandmarkDetector model
file(GLOB files "openface2/LandmarkDetector/model/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model)
endforeach()

# Move the hierarchical LandmarkDetector models
file(GLOB files "openface2/LandmarkDetector/model/model*")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model)
	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model)
endforeach()

# Move detection validation models
file(GLOB files "openface2/LandmarkDetector/model/detection_validation/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/detection_validation)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/detection_validation)
endforeach()

# Move patch experts
file(GLOB files "openface2/LandmarkDetector/model/patch_experts/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/patch_experts)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/patch_experts)
endforeach()

# Move CEN patch experts
file(GLOB files "openface2/LandmarkDetector/model/patch_experts/*.dat")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/patch_experts)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/patch_experts)
endforeach()

# Move MTCNN face detector
file(GLOB files "openface2/LandmarkDetector/model/mtcnn_detector/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/mtcnn_detector)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/mtcnn_detector)
endforeach()

# Move MTCNN face detector
file(GLOB files "openface2/LandmarkDetector/model/mtcnn_detector/*.dat")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/mtcnn_detector)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/mtcnn_detector)
endforeach()

# Move Point Distribution models
file(GLOB files "openface2/LandmarkDetector/model/pdms/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/pdms)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/pdms)
endforeach()

# Move OpenCV classifiers
file(GLOB files "lib/3rdParty/OpenCV/classifiers/*.xml")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/classifiers)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/classifiers)
endforeach()

# Move AU prediction modules
file(GLOB files "openface2/FaceAnalyser/AU_predictors/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

# Move AU prediction modules
file(GLOB files "openface2/FaceAnalyser/AU_predictors/svr*")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

# Move AU prediction modules
file(GLOB files "openface2/FaceAnalyser/AU_predictors/svm*")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

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
