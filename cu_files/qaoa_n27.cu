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
	sim.append(Simulation::U1(10.389986, 2));
	sim.append(Simulation::CX(0, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::U1(10.389986, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::U1(10.389986, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::CX(1, 3));
	sim.append(Simulation::U1(10.389986, 3));
	sim.append(Simulation::CX(1, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::U1(10.389986, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::U1(10.389986, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::CX(2, 4));
	sim.append(Simulation::U1(10.389986, 4));
	sim.append(Simulation::CX(2, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::U1(10.389986, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 5));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::U1(10.389986, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 7));
	sim.append(Simulation::CX(3, 7));
	sim.append(Simulation::U1(10.389986, 7));
	sim.append(Simulation::CX(3, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::U1(10.389986, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::U1(10.389986, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::CX(2, 8));
	sim.append(Simulation::U1(10.389986, 8));
	sim.append(Simulation::CX(2, 8));
	sim.append(Simulation::CX(3, 8));
	sim.append(Simulation::U1(10.389986, 8));
	sim.append(Simulation::CX(3, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::U1(10.389986, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::U1(10.389986, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::CX(7, 9));
	sim.append(Simulation::U1(10.389986, 9));
	sim.append(Simulation::CX(7, 9));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::U1(10.389986, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::CX(1, 10));
	sim.append(Simulation::U1(10.389986, 10));
	sim.append(Simulation::CX(1, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::U1(10.389986, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::CX(3, 10));
	sim.append(Simulation::U1(10.389986, 10));
	sim.append(Simulation::CX(3, 10));
	sim.append(Simulation::CX(5, 10));
	sim.append(Simulation::U1(10.389986, 10));
	sim.append(Simulation::CX(5, 10));
	sim.append(Simulation::CX(6, 10));
	sim.append(Simulation::U1(10.389986, 10));
	sim.append(Simulation::CX(6, 10));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::U1(10.389986, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::CX(2, 11));
	sim.append(Simulation::U1(10.389986, 11));
	sim.append(Simulation::CX(2, 11));
	sim.append(Simulation::CX(4, 11));
	sim.append(Simulation::U1(10.389986, 11));
	sim.append(Simulation::CX(4, 11));
	sim.append(Simulation::CX(5, 11));
	sim.append(Simulation::U1(10.389986, 11));
	sim.append(Simulation::CX(5, 11));
	sim.append(Simulation::CX(6, 11));
	sim.append(Simulation::U1(10.389986, 11));
	sim.append(Simulation::CX(6, 11));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 12));
	sim.append(Simulation::CX(5, 12));
	sim.append(Simulation::U1(10.389986, 12));
	sim.append(Simulation::CX(5, 12));
	sim.append(Simulation::CX(7, 12));
	sim.append(Simulation::U1(10.389986, 12));
	sim.append(Simulation::CX(7, 12));
	sim.append(Simulation::CX(8, 12));
	sim.append(Simulation::U1(10.389986, 12));
	sim.append(Simulation::CX(8, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::U1(10.389986, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::U1(10.389986, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::CX(3, 13));
	sim.append(Simulation::U1(10.389986, 13));
	sim.append(Simulation::CX(3, 13));
	sim.append(Simulation::CX(4, 13));
	sim.append(Simulation::U1(10.389986, 13));
	sim.append(Simulation::CX(4, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::U1(10.389986, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::CX(8, 13));
	sim.append(Simulation::U1(10.389986, 13));
	sim.append(Simulation::CX(8, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::U1(10.389986, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::U1(10.389986, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::CX(3, 14));
	sim.append(Simulation::U1(10.389986, 14));
	sim.append(Simulation::CX(3, 14));
	sim.append(Simulation::CX(4, 14));
	sim.append(Simulation::U1(10.389986, 14));
	sim.append(Simulation::CX(4, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::U1(10.389986, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::CX(8, 14));
	sim.append(Simulation::U1(10.389986, 14));
	sim.append(Simulation::CX(8, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::U1(10.389986, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::CX(12, 14));
	sim.append(Simulation::U1(10.389986, 14));
	sim.append(Simulation::CX(12, 14));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 15));
	sim.append(Simulation::CX(4, 15));
	sim.append(Simulation::U1(10.389986, 15));
	sim.append(Simulation::CX(4, 15));
	sim.append(Simulation::CX(6, 15));
	sim.append(Simulation::U1(10.389986, 15));
	sim.append(Simulation::CX(6, 15));
	sim.append(Simulation::CX(7, 15));
	sim.append(Simulation::U1(10.389986, 15));
	sim.append(Simulation::CX(7, 15));
	sim.append(Simulation::CX(9, 15));
	sim.append(Simulation::U1(10.389986, 15));
	sim.append(Simulation::CX(9, 15));
	sim.append(Simulation::CX(10, 15));
	sim.append(Simulation::U1(10.389986, 15));
	sim.append(Simulation::CX(10, 15));
	sim.append(Simulation::CX(11, 15));
	sim.append(Simulation::U1(10.389986, 15));
	sim.append(Simulation::CX(11, 15));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 16));
	sim.append(Simulation::CX(5, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(5, 16));
	sim.append(Simulation::CX(7, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(7, 16));
	sim.append(Simulation::CX(8, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(8, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::CX(11, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(11, 16));
	sim.append(Simulation::CX(12, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(12, 16));
	sim.append(Simulation::CX(13, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(13, 16));
	sim.append(Simulation::CX(14, 16));
	sim.append(Simulation::U1(10.389986, 16));
	sim.append(Simulation::CX(14, 16));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::CX(1, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(1, 17));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::CX(5, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(5, 17));
	sim.append(Simulation::CX(8, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(8, 17));
	sim.append(Simulation::CX(9, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(9, 17));
	sim.append(Simulation::CX(12, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(12, 17));
	sim.append(Simulation::CX(13, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(13, 17));
	sim.append(Simulation::CX(14, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(14, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::U1(10.389986, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 18));
	sim.append(Simulation::CX(1, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(1, 18));
	sim.append(Simulation::CX(2, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(2, 18));
	sim.append(Simulation::CX(3, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(3, 18));
	sim.append(Simulation::CX(4, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(4, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::CX(8, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(8, 18));
	sim.append(Simulation::CX(9, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(9, 18));
	sim.append(Simulation::CX(11, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(11, 18));
	sim.append(Simulation::CX(14, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(14, 18));
	sim.append(Simulation::CX(15, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(15, 18));
	sim.append(Simulation::CX(16, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(16, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::U1(10.389986, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 19));
	sim.append(Simulation::CX(1, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(1, 19));
	sim.append(Simulation::CX(2, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(2, 19));
	sim.append(Simulation::CX(3, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(3, 19));
	sim.append(Simulation::CX(7, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(7, 19));
	sim.append(Simulation::CX(8, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(8, 19));
	sim.append(Simulation::CX(9, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(9, 19));
	sim.append(Simulation::CX(12, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(12, 19));
	sim.append(Simulation::CX(15, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(15, 19));
	sim.append(Simulation::CX(16, 19));
	sim.append(Simulation::U1(10.389986, 19));
	sim.append(Simulation::CX(16, 19));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::CX(1, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(1, 20));
	sim.append(Simulation::CX(2, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(2, 20));
	sim.append(Simulation::CX(4, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(4, 20));
	sim.append(Simulation::CX(5, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(5, 20));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::CX(8, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(8, 20));
	sim.append(Simulation::CX(9, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(9, 20));
	sim.append(Simulation::CX(10, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(10, 20));
	sim.append(Simulation::CX(12, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(12, 20));
	sim.append(Simulation::CX(14, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(14, 20));
	sim.append(Simulation::CX(15, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(15, 20));
	sim.append(Simulation::CX(18, 20));
	sim.append(Simulation::U1(10.389986, 20));
	sim.append(Simulation::CX(18, 20));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::CX(2, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(2, 21));
	sim.append(Simulation::CX(5, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(5, 21));
	sim.append(Simulation::CX(6, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(6, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::CX(15, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(15, 21));
	sim.append(Simulation::CX(19, 21));
	sim.append(Simulation::U1(10.389986, 21));
	sim.append(Simulation::CX(19, 21));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::CX(2, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(2, 22));
	sim.append(Simulation::CX(3, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(3, 22));
	sim.append(Simulation::CX(5, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(5, 22));
	sim.append(Simulation::CX(6, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(6, 22));
	sim.append(Simulation::CX(7, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(7, 22));
	sim.append(Simulation::CX(9, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(9, 22));
	sim.append(Simulation::CX(18, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(18, 22));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 18));
	sim.append(Simulation::CX(20, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(20, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::U1(10.389986, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::CX(1, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(1, 23));
	sim.append(Simulation::CX(3, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(3, 23));
	sim.append(Simulation::CX(4, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(4, 23));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::CX(8, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(8, 23));
	sim.append(Simulation::CX(12, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(12, 23));
	sim.append(Simulation::CX(13, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(13, 23));
	sim.append(Simulation::CX(14, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(14, 23));
	sim.append(Simulation::CX(17, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(17, 23));
	sim.append(Simulation::CX(19, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(19, 23));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 19));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::U1(10.389986, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 24));
	sim.append(Simulation::CX(0, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(0, 24));
	sim.append(Simulation::CX(3, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(3, 24));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 3));
	sim.append(Simulation::CX(6, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(6, 24));
	sim.append(Simulation::CX(7, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(7, 24));
	sim.append(Simulation::CX(9, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(9, 24));
	sim.append(Simulation::CX(10, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(10, 24));
	sim.append(Simulation::CX(12, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(12, 24));
	sim.append(Simulation::CX(15, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(15, 24));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 15));
	sim.append(Simulation::CX(17, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(17, 24));
	sim.append(Simulation::CX(20, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(20, 24));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 20));
	sim.append(Simulation::CX(21, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(21, 24));
	sim.append(Simulation::CX(22, 24));
	sim.append(Simulation::U1(10.389986, 24));
	sim.append(Simulation::CX(22, 24));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 25));
	sim.append(Simulation::CX(1, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(1, 25));
	sim.append(Simulation::CX(2, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(2, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 2));
	sim.append(Simulation::CX(5, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(5, 25));
	sim.append(Simulation::CX(6, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(6, 25));
	sim.append(Simulation::CX(7, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(7, 25));
	sim.append(Simulation::CX(9, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(9, 25));
	sim.append(Simulation::CX(11, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(11, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 11));
	sim.append(Simulation::CX(12, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(12, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 12));
	sim.append(Simulation::CX(13, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(13, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 13));
	sim.append(Simulation::CX(14, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(14, 25));
	sim.append(Simulation::CX(16, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(16, 25));
	sim.append(Simulation::CX(17, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(17, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 17));
	sim.append(Simulation::CX(21, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(21, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 21));
	sim.append(Simulation::CX(22, 25));
	sim.append(Simulation::U1(10.389986, 25));
	sim.append(Simulation::CX(22, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 25));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 9));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 26));
	sim.append(Simulation::CX(0, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(0, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 0));
	sim.append(Simulation::CX(1, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(1, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 1));
	sim.append(Simulation::CX(4, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(4, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 4));
	sim.append(Simulation::CX(5, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(5, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 5));
	sim.append(Simulation::CX(6, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(6, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 6));
	sim.append(Simulation::CX(7, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(7, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 7));
	sim.append(Simulation::CX(8, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(8, 26));
	sim.append(Simulation::CX(10, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(10, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 10));
	sim.append(Simulation::CX(14, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(14, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 14));
	sim.append(Simulation::CX(16, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(16, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 16));
	sim.append(Simulation::CX(22, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(22, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 22));
	sim.append(Simulation::CX(23, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(23, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 23));
	sim.append(Simulation::CX(24, 26));
	sim.append(Simulation::U1(10.389986, 26));
	sim.append(Simulation::CX(24, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 24));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 26));
	sim.append(Simulation::U3(2.4837228, -1.5707963267948966, 1.5707963267948966, 8));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=27;
	int n_gpus=2;
	int chunkSize=22;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
