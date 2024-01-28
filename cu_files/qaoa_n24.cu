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
	sim.append(Simulation::U1(6.5827273, 2));
	sim.append(Simulation::CX(0, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::U1(6.5827273, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::U1(6.5827273, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::U1(6.5827273, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::U1(6.5827273, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::CX(1, 4));
	sim.append(Simulation::U1(6.5827273, 4));
	sim.append(Simulation::CX(1, 4));
	sim.append(Simulation::CX(2, 4));
	sim.append(Simulation::U1(6.5827273, 4));
	sim.append(Simulation::CX(2, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::U1(6.5827273, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 5));
	sim.append(Simulation::CX(1, 5));
	sim.append(Simulation::U1(6.5827273, 5));
	sim.append(Simulation::CX(1, 5));
	sim.append(Simulation::CX(2, 5));
	sim.append(Simulation::U1(6.5827273, 5));
	sim.append(Simulation::CX(2, 5));
	sim.append(Simulation::CX(3, 5));
	sim.append(Simulation::U1(6.5827273, 5));
	sim.append(Simulation::CX(3, 5));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 6));
	sim.append(Simulation::CX(1, 6));
	sim.append(Simulation::U1(6.5827273, 6));
	sim.append(Simulation::CX(1, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::U1(6.5827273, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 7));
	sim.append(Simulation::CX(2, 7));
	sim.append(Simulation::U1(6.5827273, 7));
	sim.append(Simulation::CX(2, 7));
	sim.append(Simulation::CX(3, 7));
	sim.append(Simulation::U1(6.5827273, 7));
	sim.append(Simulation::CX(3, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::U1(6.5827273, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::CX(5, 7));
	sim.append(Simulation::U1(6.5827273, 7));
	sim.append(Simulation::CX(5, 7));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::U1(6.5827273, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::CX(2, 8));
	sim.append(Simulation::U1(6.5827273, 8));
	sim.append(Simulation::CX(2, 8));
	sim.append(Simulation::CX(3, 8));
	sim.append(Simulation::U1(6.5827273, 8));
	sim.append(Simulation::CX(3, 8));
	sim.append(Simulation::CX(4, 8));
	sim.append(Simulation::U1(6.5827273, 8));
	sim.append(Simulation::CX(4, 8));
	sim.append(Simulation::CX(5, 8));
	sim.append(Simulation::U1(6.5827273, 8));
	sim.append(Simulation::CX(5, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::U1(6.5827273, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::U1(6.5827273, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::U1(6.5827273, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::CX(2, 9));
	sim.append(Simulation::U1(6.5827273, 9));
	sim.append(Simulation::CX(2, 9));
	sim.append(Simulation::CX(4, 9));
	sim.append(Simulation::U1(6.5827273, 9));
	sim.append(Simulation::CX(4, 9));
	sim.append(Simulation::CX(5, 9));
	sim.append(Simulation::U1(6.5827273, 9));
	sim.append(Simulation::CX(5, 9));
	sim.append(Simulation::CX(6, 9));
	sim.append(Simulation::U1(6.5827273, 9));
	sim.append(Simulation::CX(6, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::U1(6.5827273, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::U1(6.5827273, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::U1(6.5827273, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::CX(6, 10));
	sim.append(Simulation::U1(6.5827273, 10));
	sim.append(Simulation::CX(6, 10));
	sim.append(Simulation::CX(7, 10));
	sim.append(Simulation::U1(6.5827273, 10));
	sim.append(Simulation::CX(7, 10));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::U1(6.5827273, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::CX(3, 11));
	sim.append(Simulation::U1(6.5827273, 11));
	sim.append(Simulation::CX(3, 11));
	sim.append(Simulation::CX(8, 11));
	sim.append(Simulation::U1(6.5827273, 11));
	sim.append(Simulation::CX(8, 11));
	sim.append(Simulation::CX(9, 11));
	sim.append(Simulation::U1(6.5827273, 11));
	sim.append(Simulation::CX(9, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::U1(6.5827273, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 12));
	sim.append(Simulation::CX(4, 12));
	sim.append(Simulation::U1(6.5827273, 12));
	sim.append(Simulation::CX(4, 12));
	sim.append(Simulation::CX(5, 12));
	sim.append(Simulation::U1(6.5827273, 12));
	sim.append(Simulation::CX(5, 12));
	sim.append(Simulation::CX(7, 12));
	sim.append(Simulation::U1(6.5827273, 12));
	sim.append(Simulation::CX(7, 12));
	sim.append(Simulation::CX(8, 12));
	sim.append(Simulation::U1(6.5827273, 12));
	sim.append(Simulation::CX(8, 12));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::CX(1, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(1, 13));
	sim.append(Simulation::CX(3, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(3, 13));
	sim.append(Simulation::CX(4, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(4, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::CX(8, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(8, 13));
	sim.append(Simulation::CX(10, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(10, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::U1(6.5827273, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::U1(6.5827273, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::CX(2, 14));
	sim.append(Simulation::U1(6.5827273, 14));
	sim.append(Simulation::CX(2, 14));
	sim.append(Simulation::CX(4, 14));
	sim.append(Simulation::U1(6.5827273, 14));
	sim.append(Simulation::CX(4, 14));
	sim.append(Simulation::CX(5, 14));
	sim.append(Simulation::U1(6.5827273, 14));
	sim.append(Simulation::CX(5, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::U1(6.5827273, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::CX(9, 14));
	sim.append(Simulation::U1(6.5827273, 14));
	sim.append(Simulation::CX(9, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::U1(6.5827273, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 15));
	sim.append(Simulation::CX(3, 15));
	sim.append(Simulation::U1(6.5827273, 15));
	sim.append(Simulation::CX(3, 15));
	sim.append(Simulation::CX(6, 15));
	sim.append(Simulation::U1(6.5827273, 15));
	sim.append(Simulation::CX(6, 15));
	sim.append(Simulation::CX(8, 15));
	sim.append(Simulation::U1(6.5827273, 15));
	sim.append(Simulation::CX(8, 15));
	sim.append(Simulation::CX(9, 15));
	sim.append(Simulation::U1(6.5827273, 15));
	sim.append(Simulation::CX(9, 15));
	sim.append(Simulation::CX(10, 15));
	sim.append(Simulation::U1(6.5827273, 15));
	sim.append(Simulation::CX(10, 15));
	sim.append(Simulation::CX(12, 15));
	sim.append(Simulation::U1(6.5827273, 15));
	sim.append(Simulation::CX(12, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::U1(6.5827273, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 16));
	sim.append(Simulation::CX(2, 16));
	sim.append(Simulation::U1(6.5827273, 16));
	sim.append(Simulation::CX(2, 16));
	sim.append(Simulation::CX(3, 16));
	sim.append(Simulation::U1(6.5827273, 16));
	sim.append(Simulation::CX(3, 16));
	sim.append(Simulation::CX(7, 16));
	sim.append(Simulation::U1(6.5827273, 16));
	sim.append(Simulation::CX(7, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::U1(6.5827273, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::CX(10, 16));
	sim.append(Simulation::U1(6.5827273, 16));
	sim.append(Simulation::CX(10, 16));
	sim.append(Simulation::CX(12, 16));
	sim.append(Simulation::U1(6.5827273, 16));
	sim.append(Simulation::CX(12, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::U1(6.5827273, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::CX(2, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(2, 17));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 2));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::CX(5, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(5, 17));
	sim.append(Simulation::CX(7, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(7, 17));
	sim.append(Simulation::CX(9, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(9, 17));
	sim.append(Simulation::CX(10, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(10, 17));
	sim.append(Simulation::CX(11, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(11, 17));
	sim.append(Simulation::CX(13, 17));
	sim.append(Simulation::U1(6.5827273, 17));
	sim.append(Simulation::CX(13, 17));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::U1(6.5827273, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::CX(8, 18));
	sim.append(Simulation::U1(6.5827273, 18));
	sim.append(Simulation::CX(8, 18));
	sim.append(Simulation::CX(9, 18));
	sim.append(Simulation::U1(6.5827273, 18));
	sim.append(Simulation::CX(9, 18));
	sim.append(Simulation::CX(10, 18));
	sim.append(Simulation::U1(6.5827273, 18));
	sim.append(Simulation::CX(10, 18));
	sim.append(Simulation::CX(11, 18));
	sim.append(Simulation::U1(6.5827273, 18));
	sim.append(Simulation::CX(11, 18));
	sim.append(Simulation::CX(14, 18));
	sim.append(Simulation::U1(6.5827273, 18));
	sim.append(Simulation::CX(14, 18));
	sim.append(Simulation::CX(16, 18));
	sim.append(Simulation::U1(6.5827273, 18));
	sim.append(Simulation::CX(16, 18));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 19));
	sim.append(Simulation::CX(3, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(3, 19));
	sim.append(Simulation::CX(4, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(4, 19));
	sim.append(Simulation::CX(6, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(6, 19));
	sim.append(Simulation::CX(8, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(8, 19));
	sim.append(Simulation::CX(11, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(11, 19));
	sim.append(Simulation::CX(14, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(14, 19));
	sim.append(Simulation::CX(16, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(16, 19));
	sim.append(Simulation::CX(17, 19));
	sim.append(Simulation::U1(6.5827273, 19));
	sim.append(Simulation::CX(17, 19));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::CX(1, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(1, 20));
	sim.append(Simulation::CX(5, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(5, 20));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 5));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::CX(7, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(7, 20));
	sim.append(Simulation::CX(10, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(10, 20));
	sim.append(Simulation::CX(11, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(11, 20));
	sim.append(Simulation::CX(15, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(15, 20));
	sim.append(Simulation::CX(17, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(17, 20));
	sim.append(Simulation::CX(18, 20));
	sim.append(Simulation::U1(6.5827273, 20));
	sim.append(Simulation::CX(18, 20));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::CX(12, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(12, 21));
	sim.append(Simulation::CX(13, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(13, 21));
	sim.append(Simulation::CX(14, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(14, 21));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 14));
	sim.append(Simulation::CX(16, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(16, 21));
	sim.append(Simulation::CX(17, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(17, 21));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 17));
	sim.append(Simulation::CX(18, 21));
	sim.append(Simulation::U1(6.5827273, 21));
	sim.append(Simulation::CX(18, 21));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 18));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::CX(3, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(3, 22));
	sim.append(Simulation::CX(7, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(7, 22));
	sim.append(Simulation::CX(8, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(8, 22));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 8));
	sim.append(Simulation::CX(9, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(9, 22));
	sim.append(Simulation::CX(11, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(11, 22));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 11));
	sim.append(Simulation::CX(12, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(12, 22));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 12));
	sim.append(Simulation::CX(15, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(15, 22));
	sim.append(Simulation::CX(16, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(16, 22));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 16));
	sim.append(Simulation::CX(20, 22));
	sim.append(Simulation::U1(6.5827273, 22));
	sim.append(Simulation::CX(20, 22));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 20));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 0));
	sim.append(Simulation::CX(1, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(1, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 1));
	sim.append(Simulation::CX(3, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(3, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 3));
	sim.append(Simulation::CX(4, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(4, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 4));
	sim.append(Simulation::CX(6, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(6, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 6));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 7));
	sim.append(Simulation::CX(9, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(9, 23));
	sim.append(Simulation::CX(10, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(10, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 10));
	sim.append(Simulation::CX(13, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(13, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 13));
	sim.append(Simulation::CX(15, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(15, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 15));
	sim.append(Simulation::CX(19, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(19, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 19));
	sim.append(Simulation::CX(21, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(21, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 21));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::U1(6.5827273, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 22));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 23));
	sim.append(Simulation::U3(1.0091405, -1.5707963267948966, 1.5707963267948966, 9));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=24;
	int n_gpus=2;
	int chunkSize=19;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
