set(CalcElementMassMatrix_SRCS massMatrix.cpp TPZMatMassMatrix.cpp)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/unitcube.msh ${CMAKE_CURRENT_BINARY_DIR}/unitcube.msh COPYONLY)

add_executable(CalcElementMassMatrix ${CalcElementMassMatrix_SRCS} )

target_link_libraries(CalcElementMassMatrix pz blaze_target)
target_include_directories(CalcElementMassMatrix PUBLIC ${PZ_INCLUDE_DIRS})
