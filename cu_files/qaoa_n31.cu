#include <stdio.h>
#include "bmlqsim.cuh"
//Use the SVSim namespace to enable C++/CUDA APIs
using namespace SVSim;

void prepare_circuit(Simulation &sim)
{
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 0));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 1));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 2));
	sim.append(Simulation::CX(0, 2));
	sim.append(Simulation::U1(9.0564946, 2));
	sim.append(Simulation::CX(0, 2));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::U1(9.0564946, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::U1(9.0564946, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::U1(9.0564946, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 5));
	sim.append(Simulation::CX(3, 5));
	sim.append(Simulation::U1(9.0564946, 5));
	sim.append(Simulation::CX(3, 5));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 6));
	sim.append(Simulation::CX(1, 6));
	sim.append(Simulation::U1(9.0564946, 6));
	sim.append(Simulation::CX(1, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::U1(9.0564946, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::U1(9.0564946, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 7));
	sim.append(Simulation::CX(3, 7));
	sim.append(Simulation::U1(9.0564946, 7));
	sim.append(Simulation::CX(3, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::U1(9.0564946, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::U1(9.0564946, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::U1(9.0564946, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::CX(5, 8));
	sim.append(Simulation::U1(9.0564946, 8));
	sim.append(Simulation::CX(5, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::U1(9.0564946, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::U1(9.0564946, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::U1(9.0564946, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::CX(7, 9));
	sim.append(Simulation::U1(9.0564946, 9));
	sim.append(Simulation::CX(7, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::U1(9.0564946, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::CX(3, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(3, 10));
	sim.append(Simulation::CX(4, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(4, 10));
	sim.append(Simulation::CX(5, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(5, 10));
	sim.append(Simulation::CX(6, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(6, 10));
	sim.append(Simulation::CX(7, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(7, 10));
	sim.append(Simulation::CX(8, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(8, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::U1(9.0564946, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::U1(9.0564946, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::CX(2, 11));
	sim.append(Simulation::U1(9.0564946, 11));
	sim.append(Simulation::CX(2, 11));
	sim.append(Simulation::CX(3, 11));
	sim.append(Simulation::U1(9.0564946, 11));
	sim.append(Simulation::CX(3, 11));
	sim.append(Simulation::CX(5, 11));
	sim.append(Simulation::U1(9.0564946, 11));
	sim.append(Simulation::CX(5, 11));
	sim.append(Simulation::CX(6, 11));
	sim.append(Simulation::U1(9.0564946, 11));
	sim.append(Simulation::CX(6, 11));
	sim.append(Simulation::CX(7, 11));
	sim.append(Simulation::U1(9.0564946, 11));
	sim.append(Simulation::CX(7, 11));
	sim.append(Simulation::CX(9, 11));
	sim.append(Simulation::U1(9.0564946, 11));
	sim.append(Simulation::CX(9, 11));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 12));
	sim.append(Simulation::CX(2, 12));
	sim.append(Simulation::U1(9.0564946, 12));
	sim.append(Simulation::CX(2, 12));
	sim.append(Simulation::CX(3, 12));
	sim.append(Simulation::U1(9.0564946, 12));
	sim.append(Simulation::CX(3, 12));
	sim.append(Simulation::CX(10, 12));
	sim.append(Simulation::U1(9.0564946, 12));
	sim.append(Simulation::CX(10, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::U1(9.0564946, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::U1(9.0564946, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::CX(1, 13));
	sim.append(Simulation::U1(9.0564946, 13));
	sim.append(Simulation::CX(1, 13));
	sim.append(Simulation::CX(2, 13));
	sim.append(Simulation::U1(9.0564946, 13));
	sim.append(Simulation::CX(2, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::U1(9.0564946, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::CX(9, 13));
	sim.append(Simulation::U1(9.0564946, 13));
	sim.append(Simulation::CX(9, 13));
	sim.append(Simulation::CX(10, 13));
	sim.append(Simulation::U1(9.0564946, 13));
	sim.append(Simulation::CX(10, 13));
	sim.append(Simulation::CX(11, 13));
	sim.append(Simulation::U1(9.0564946, 13));
	sim.append(Simulation::CX(11, 13));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::U1(9.0564946, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::CX(1, 14));
	sim.append(Simulation::U1(9.0564946, 14));
	sim.append(Simulation::CX(1, 14));
	sim.append(Simulation::CX(2, 14));
	sim.append(Simulation::U1(9.0564946, 14));
	sim.append(Simulation::CX(2, 14));
	sim.append(Simulation::CX(5, 14));
	sim.append(Simulation::U1(9.0564946, 14));
	sim.append(Simulation::CX(5, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::U1(9.0564946, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::U1(9.0564946, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::U1(9.0564946, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 15));
	sim.append(Simulation::CX(1, 15));
	sim.append(Simulation::U1(9.0564946, 15));
	sim.append(Simulation::CX(1, 15));
	sim.append(Simulation::CX(4, 15));
	sim.append(Simulation::U1(9.0564946, 15));
	sim.append(Simulation::CX(4, 15));
	sim.append(Simulation::CX(5, 15));
	sim.append(Simulation::U1(9.0564946, 15));
	sim.append(Simulation::CX(5, 15));
	sim.append(Simulation::CX(8, 15));
	sim.append(Simulation::U1(9.0564946, 15));
	sim.append(Simulation::CX(8, 15));
	sim.append(Simulation::CX(11, 15));
	sim.append(Simulation::U1(9.0564946, 15));
	sim.append(Simulation::CX(11, 15));
	sim.append(Simulation::CX(12, 15));
	sim.append(Simulation::U1(9.0564946, 15));
	sim.append(Simulation::CX(12, 15));
	sim.append(Simulation::CX(13, 15));
	sim.append(Simulation::U1(9.0564946, 15));
	sim.append(Simulation::CX(13, 15));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 16));
	sim.append(Simulation::CX(1, 16));
	sim.append(Simulation::U1(9.0564946, 16));
	sim.append(Simulation::CX(1, 16));
	sim.append(Simulation::CX(4, 16));
	sim.append(Simulation::U1(9.0564946, 16));
	sim.append(Simulation::CX(4, 16));
	sim.append(Simulation::CX(7, 16));
	sim.append(Simulation::U1(9.0564946, 16));
	sim.append(Simulation::CX(7, 16));
	sim.append(Simulation::CX(8, 16));
	sim.append(Simulation::U1(9.0564946, 16));
	sim.append(Simulation::CX(8, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::U1(9.0564946, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::CX(12, 16));
	sim.append(Simulation::U1(9.0564946, 16));
	sim.append(Simulation::CX(12, 16));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::CX(1, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(1, 17));
	sim.append(Simulation::CX(2, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(2, 17));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::CX(4, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(4, 17));
	sim.append(Simulation::CX(6, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(6, 17));
	sim.append(Simulation::CX(12, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(12, 17));
	sim.append(Simulation::CX(14, 17));
	sim.append(Simulation::U1(9.0564946, 17));
	sim.append(Simulation::CX(14, 17));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 18));
	sim.append(Simulation::CX(1, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(1, 18));
	sim.append(Simulation::CX(5, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(5, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::CX(10, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(10, 18));
	sim.append(Simulation::CX(11, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(11, 18));
	sim.append(Simulation::CX(13, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(13, 18));
	sim.append(Simulation::CX(15, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(15, 18));
	sim.append(Simulation::CX(16, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(16, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::U1(9.0564946, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 19));
	sim.append(Simulation::CX(1, 19));
	sim.append(Simulation::U1(9.0564946, 19));
	sim.append(Simulation::CX(1, 19));
	sim.append(Simulation::CX(2, 19));
	sim.append(Simulation::U1(9.0564946, 19));
	sim.append(Simulation::CX(2, 19));
	sim.append(Simulation::CX(6, 19));
	sim.append(Simulation::U1(9.0564946, 19));
	sim.append(Simulation::CX(6, 19));
	sim.append(Simulation::CX(10, 19));
	sim.append(Simulation::U1(9.0564946, 19));
	sim.append(Simulation::CX(10, 19));
	sim.append(Simulation::CX(14, 19));
	sim.append(Simulation::U1(9.0564946, 19));
	sim.append(Simulation::CX(14, 19));
	sim.append(Simulation::CX(17, 19));
	sim.append(Simulation::U1(9.0564946, 19));
	sim.append(Simulation::CX(17, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::U1(9.0564946, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::CX(5, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(5, 20));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::CX(9, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(9, 20));
	sim.append(Simulation::CX(11, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(11, 20));
	sim.append(Simulation::CX(13, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(13, 20));
	sim.append(Simulation::CX(15, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(15, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::U1(9.0564946, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::CX(2, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(2, 21));
	sim.append(Simulation::CX(3, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(3, 21));
	sim.append(Simulation::CX(4, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(4, 21));
	sim.append(Simulation::CX(5, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(5, 21));
	sim.append(Simulation::CX(6, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(6, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::CX(16, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(16, 21));
	sim.append(Simulation::CX(17, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(17, 21));
	sim.append(Simulation::CX(19, 21));
	sim.append(Simulation::U1(9.0564946, 21));
	sim.append(Simulation::CX(19, 21));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::CX(2, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(2, 22));
	sim.append(Simulation::CX(4, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(4, 22));
	sim.append(Simulation::CX(5, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(5, 22));
	sim.append(Simulation::CX(7, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(7, 22));
	sim.append(Simulation::CX(9, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(9, 22));
	sim.append(Simulation::CX(14, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(14, 22));
	sim.append(Simulation::CX(17, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(17, 22));
	sim.append(Simulation::CX(19, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(19, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::U1(9.0564946, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::CX(1, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(1, 23));
	sim.append(Simulation::CX(2, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(2, 23));
	sim.append(Simulation::CX(3, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(3, 23));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::CX(9, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(9, 23));
	sim.append(Simulation::CX(12, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(12, 23));
	sim.append(Simulation::CX(16, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(16, 23));
	sim.append(Simulation::CX(19, 23));
	sim.append(Simulation::U1(9.0564946, 23));
	sim.append(Simulation::CX(19, 23));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 24));
	sim.append(Simulation::CX(0, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(0, 24));
	sim.append(Simulation::CX(1, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(1, 24));
	sim.append(Simulation::CX(3, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(3, 24));
	sim.append(Simulation::CX(5, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(5, 24));
	sim.append(Simulation::CX(7, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(7, 24));
	sim.append(Simulation::CX(11, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(11, 24));
	sim.append(Simulation::CX(12, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(12, 24));
	sim.append(Simulation::CX(14, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(14, 24));
	sim.append(Simulation::CX(15, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(15, 24));
	sim.append(Simulation::CX(18, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(18, 24));
	sim.append(Simulation::CX(20, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(20, 24));
	sim.append(Simulation::CX(21, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(21, 24));
	sim.append(Simulation::CX(22, 24));
	sim.append(Simulation::U1(9.0564946, 24));
	sim.append(Simulation::CX(22, 24));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 24));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 25));
	sim.append(Simulation::CX(2, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(2, 25));
	sim.append(Simulation::CX(3, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(3, 25));
	sim.append(Simulation::CX(4, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(4, 25));
	sim.append(Simulation::CX(5, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(5, 25));
	sim.append(Simulation::CX(6, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(6, 25));
	sim.append(Simulation::CX(7, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(7, 25));
	sim.append(Simulation::CX(8, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(8, 25));
	sim.append(Simulation::CX(9, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(9, 25));
	sim.append(Simulation::CX(10, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(10, 25));
	sim.append(Simulation::CX(11, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(11, 25));
	sim.append(Simulation::CX(12, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(12, 25));
	sim.append(Simulation::CX(14, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(14, 25));
	sim.append(Simulation::CX(16, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(16, 25));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 16));
	sim.append(Simulation::CX(17, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(17, 25));
	sim.append(Simulation::CX(20, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(20, 25));
	sim.append(Simulation::CX(22, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(22, 25));
	sim.append(Simulation::CX(23, 25));
	sim.append(Simulation::U1(9.0564946, 25));
	sim.append(Simulation::CX(23, 25));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 23));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 26));
	sim.append(Simulation::CX(0, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(0, 26));
	sim.append(Simulation::CX(4, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(4, 26));
	sim.append(Simulation::CX(5, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(5, 26));
	sim.append(Simulation::CX(6, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(6, 26));
	sim.append(Simulation::CX(7, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(7, 26));
	sim.append(Simulation::CX(9, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(9, 26));
	sim.append(Simulation::CX(11, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(11, 26));
	sim.append(Simulation::CX(13, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(13, 26));
	sim.append(Simulation::CX(15, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(15, 26));
	sim.append(Simulation::CX(17, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(17, 26));
	sim.append(Simulation::CX(18, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(18, 26));
	sim.append(Simulation::CX(19, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(19, 26));
	sim.append(Simulation::CX(20, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(20, 26));
	sim.append(Simulation::CX(21, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(21, 26));
	sim.append(Simulation::CX(22, 26));
	sim.append(Simulation::U1(9.0564946, 26));
	sim.append(Simulation::CX(22, 26));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 26));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 27));
	sim.append(Simulation::CX(0, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(0, 27));
	sim.append(Simulation::CX(4, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(4, 27));
	sim.append(Simulation::CX(6, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(6, 27));
	sim.append(Simulation::CX(7, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(7, 27));
	sim.append(Simulation::CX(8, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(8, 27));
	sim.append(Simulation::CX(10, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(10, 27));
	sim.append(Simulation::CX(11, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(11, 27));
	sim.append(Simulation::CX(14, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(14, 27));
	sim.append(Simulation::CX(17, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(17, 27));
	sim.append(Simulation::CX(18, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(18, 27));
	sim.append(Simulation::CX(19, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(19, 27));
	sim.append(Simulation::CX(20, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(20, 27));
	sim.append(Simulation::CX(21, 27));
	sim.append(Simulation::U1(9.0564946, 27));
	sim.append(Simulation::CX(21, 27));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 28));
	sim.append(Simulation::CX(0, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(0, 28));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 0));
	sim.append(Simulation::CX(1, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(1, 28));
	sim.append(Simulation::CX(2, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(2, 28));
	sim.append(Simulation::CX(3, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(3, 28));
	sim.append(Simulation::CX(5, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(5, 28));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 5));
	sim.append(Simulation::CX(8, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(8, 28));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 8));
	sim.append(Simulation::CX(9, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(9, 28));
	sim.append(Simulation::CX(13, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(13, 28));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 13));
	sim.append(Simulation::CX(14, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(14, 28));
	sim.append(Simulation::CX(15, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(15, 28));
	sim.append(Simulation::CX(17, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(17, 28));
	sim.append(Simulation::CX(18, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(18, 28));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 18));
	sim.append(Simulation::CX(20, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(20, 28));
	sim.append(Simulation::CX(25, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(25, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::U1(9.0564946, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 29));
	sim.append(Simulation::CX(2, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(2, 29));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 2));
	sim.append(Simulation::CX(6, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(6, 29));
	sim.append(Simulation::CX(7, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(7, 29));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 7));
	sim.append(Simulation::CX(9, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(9, 29));
	sim.append(Simulation::CX(14, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(14, 29));
	sim.append(Simulation::CX(15, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(15, 29));
	sim.append(Simulation::CX(17, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(17, 29));
	sim.append(Simulation::CX(21, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(21, 29));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 21));
	sim.append(Simulation::CX(22, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(22, 29));
	sim.append(Simulation::CX(25, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(25, 29));
	sim.append(Simulation::CX(27, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(27, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::U1(9.0564946, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 29));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 9));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 30));
	sim.append(Simulation::CX(1, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(1, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 1));
	sim.append(Simulation::CX(3, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(3, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 3));
	sim.append(Simulation::CX(4, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(4, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 4));
	sim.append(Simulation::CX(6, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(6, 30));
	sim.append(Simulation::CX(10, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(10, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 10));
	sim.append(Simulation::CX(11, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(11, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 11));
	sim.append(Simulation::CX(12, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(12, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 12));
	sim.append(Simulation::CX(14, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(14, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 14));
	sim.append(Simulation::CX(15, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(15, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 15));
	sim.append(Simulation::CX(17, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(17, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 17));
	sim.append(Simulation::CX(19, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(19, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 19));
	sim.append(Simulation::CX(20, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(20, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 20));
	sim.append(Simulation::CX(22, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(22, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 22));
	sim.append(Simulation::CX(25, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(25, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 25));
	sim.append(Simulation::CX(27, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(27, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 27));
	sim.append(Simulation::CX(28, 30));
	sim.append(Simulation::U1(9.0564946, 30));
	sim.append(Simulation::CX(28, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 28));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 30));
	sim.append(Simulation::U3(4.7115632, -1.5707963267948966, 1.5707963267948966, 6));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=31;
	int n_gpus=2;
	int chunkSize=25;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
