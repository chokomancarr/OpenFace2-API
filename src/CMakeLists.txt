# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(GetLandmarks GetLandmarks.cpp)
target_link_libraries(GetLandmarks LandmarkDetector)
target_link_libraries(GetLandmarks FaceAnalyser)
target_link_libraries(GetLandmarks GazeAnalyser)
target_link_libraries(GetLandmarks Utilities)

install (TARGETS GetLandmarks DESTINATION bin)