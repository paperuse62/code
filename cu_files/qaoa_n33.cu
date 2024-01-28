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
	sim.append(Simulation::U1(4.5683703, 2));
	sim.append(Simulation::CX(0, 2));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::U1(4.5683703, 3));
	sim.append(Simulation::CX(0, 3));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::U1(4.5683703, 4));
	sim.append(Simulation::CX(0, 4));
	sim.append(Simulation::CX(1, 4));
	sim.append(Simulation::U1(4.5683703, 4));
	sim.append(Simulation::CX(1, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::U1(4.5683703, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 5));
	sim.append(Simulation::CX(3, 5));
	sim.append(Simulation::U1(4.5683703, 5));
	sim.append(Simulation::CX(3, 5));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 6));
	sim.append(Simulation::CX(2, 6));
	sim.append(Simulation::U1(4.5683703, 6));
	sim.append(Simulation::CX(2, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::U1(4.5683703, 6));
	sim.append(Simulation::CX(3, 6));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 7));
	sim.append(Simulation::CX(2, 7));
	sim.append(Simulation::U1(4.5683703, 7));
	sim.append(Simulation::CX(2, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::U1(4.5683703, 7));
	sim.append(Simulation::CX(4, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::U1(4.5683703, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::U1(4.5683703, 8));
	sim.append(Simulation::CX(0, 8));
	sim.append(Simulation::CX(2, 8));
	sim.append(Simulation::U1(4.5683703, 8));
	sim.append(Simulation::CX(2, 8));
	sim.append(Simulation::CX(4, 8));
	sim.append(Simulation::U1(4.5683703, 8));
	sim.append(Simulation::CX(4, 8));
	sim.append(Simulation::CX(5, 8));
	sim.append(Simulation::U1(4.5683703, 8));
	sim.append(Simulation::CX(5, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::U1(4.5683703, 8));
	sim.append(Simulation::CX(6, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::U1(4.5683703, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::U1(4.5683703, 9));
	sim.append(Simulation::CX(0, 9));
	sim.append(Simulation::CX(2, 9));
	sim.append(Simulation::U1(4.5683703, 9));
	sim.append(Simulation::CX(2, 9));
	sim.append(Simulation::CX(4, 9));
	sim.append(Simulation::U1(4.5683703, 9));
	sim.append(Simulation::CX(4, 9));
	sim.append(Simulation::CX(6, 9));
	sim.append(Simulation::U1(4.5683703, 9));
	sim.append(Simulation::CX(6, 9));
	sim.append(Simulation::CX(7, 9));
	sim.append(Simulation::U1(4.5683703, 9));
	sim.append(Simulation::CX(7, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::U1(4.5683703, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::U1(4.5683703, 10));
	sim.append(Simulation::CX(0, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::U1(4.5683703, 10));
	sim.append(Simulation::CX(2, 10));
	sim.append(Simulation::CX(5, 10));
	sim.append(Simulation::U1(4.5683703, 10));
	sim.append(Simulation::CX(5, 10));
	sim.append(Simulation::CX(7, 10));
	sim.append(Simulation::U1(4.5683703, 10));
	sim.append(Simulation::CX(7, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::U1(4.5683703, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(0, 11));
	sim.append(Simulation::CX(1, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(1, 11));
	sim.append(Simulation::CX(3, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(3, 11));
	sim.append(Simulation::CX(5, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(5, 11));
	sim.append(Simulation::CX(7, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(7, 11));
	sim.append(Simulation::CX(8, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(8, 11));
	sim.append(Simulation::CX(9, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(9, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::U1(4.5683703, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 12));
	sim.append(Simulation::CX(1, 12));
	sim.append(Simulation::U1(4.5683703, 12));
	sim.append(Simulation::CX(1, 12));
	sim.append(Simulation::CX(5, 12));
	sim.append(Simulation::U1(4.5683703, 12));
	sim.append(Simulation::CX(5, 12));
	sim.append(Simulation::CX(7, 12));
	sim.append(Simulation::U1(4.5683703, 12));
	sim.append(Simulation::CX(7, 12));
	sim.append(Simulation::CX(8, 12));
	sim.append(Simulation::U1(4.5683703, 12));
	sim.append(Simulation::CX(8, 12));
	sim.append(Simulation::CX(10, 12));
	sim.append(Simulation::U1(4.5683703, 12));
	sim.append(Simulation::CX(10, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::U1(4.5683703, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(0, 13));
	sim.append(Simulation::CX(1, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(1, 13));
	sim.append(Simulation::CX(2, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(2, 13));
	sim.append(Simulation::CX(4, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(4, 13));
	sim.append(Simulation::CX(6, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(6, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(7, 13));
	sim.append(Simulation::CX(11, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(11, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::U1(4.5683703, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(0, 14));
	sim.append(Simulation::CX(1, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(1, 14));
	sim.append(Simulation::CX(4, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(4, 14));
	sim.append(Simulation::CX(5, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(5, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(6, 14));
	sim.append(Simulation::CX(9, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(9, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(10, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::U1(4.5683703, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 15));
	sim.append(Simulation::CX(1, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(1, 15));
	sim.append(Simulation::CX(2, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(2, 15));
	sim.append(Simulation::CX(3, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(3, 15));
	sim.append(Simulation::CX(5, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(5, 15));
	sim.append(Simulation::CX(6, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(6, 15));
	sim.append(Simulation::CX(7, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(7, 15));
	sim.append(Simulation::CX(8, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(8, 15));
	sim.append(Simulation::CX(9, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(9, 15));
	sim.append(Simulation::CX(13, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(13, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::U1(4.5683703, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 16));
	sim.append(Simulation::CX(1, 16));
	sim.append(Simulation::U1(4.5683703, 16));
	sim.append(Simulation::CX(1, 16));
	sim.append(Simulation::CX(5, 16));
	sim.append(Simulation::U1(4.5683703, 16));
	sim.append(Simulation::CX(5, 16));
	sim.append(Simulation::CX(8, 16));
	sim.append(Simulation::U1(4.5683703, 16));
	sim.append(Simulation::CX(8, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::U1(4.5683703, 16));
	sim.append(Simulation::CX(9, 16));
	sim.append(Simulation::CX(14, 16));
	sim.append(Simulation::U1(4.5683703, 16));
	sim.append(Simulation::CX(14, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::U1(4.5683703, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(0, 17));
	sim.append(Simulation::CX(1, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(1, 17));
	sim.append(Simulation::CX(2, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(2, 17));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(3, 17));
	sim.append(Simulation::CX(4, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(4, 17));
	sim.append(Simulation::CX(6, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(6, 17));
	sim.append(Simulation::CX(7, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(7, 17));
	sim.append(Simulation::CX(10, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(10, 17));
	sim.append(Simulation::CX(13, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(13, 17));
	sim.append(Simulation::CX(14, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(14, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::U1(4.5683703, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 18));
	sim.append(Simulation::CX(2, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(2, 18));
	sim.append(Simulation::CX(3, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(3, 18));
	sim.append(Simulation::CX(4, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(4, 18));
	sim.append(Simulation::CX(5, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(5, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(6, 18));
	sim.append(Simulation::CX(7, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(7, 18));
	sim.append(Simulation::CX(8, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(8, 18));
	sim.append(Simulation::CX(13, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(13, 18));
	sim.append(Simulation::CX(15, 18));
	sim.append(Simulation::U1(4.5683703, 18));
	sim.append(Simulation::CX(15, 18));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 19));
	sim.append(Simulation::CX(1, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(1, 19));
	sim.append(Simulation::CX(2, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(2, 19));
	sim.append(Simulation::CX(3, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(3, 19));
	sim.append(Simulation::CX(4, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(4, 19));
	sim.append(Simulation::CX(6, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(6, 19));
	sim.append(Simulation::CX(10, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(10, 19));
	sim.append(Simulation::CX(11, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(11, 19));
	sim.append(Simulation::CX(12, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(12, 19));
	sim.append(Simulation::CX(13, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(13, 19));
	sim.append(Simulation::CX(14, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(14, 19));
	sim.append(Simulation::CX(15, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(15, 19));
	sim.append(Simulation::CX(16, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(16, 19));
	sim.append(Simulation::CX(17, 19));
	sim.append(Simulation::U1(4.5683703, 19));
	sim.append(Simulation::CX(17, 19));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(0, 20));
	sim.append(Simulation::CX(1, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(1, 20));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(6, 20));
	sim.append(Simulation::CX(9, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(9, 20));
	sim.append(Simulation::CX(10, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(10, 20));
	sim.append(Simulation::CX(11, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(11, 20));
	sim.append(Simulation::CX(13, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(13, 20));
	sim.append(Simulation::CX(16, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(16, 20));
	sim.append(Simulation::CX(18, 20));
	sim.append(Simulation::U1(4.5683703, 20));
	sim.append(Simulation::CX(18, 20));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(1, 21));
	sim.append(Simulation::CX(2, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(2, 21));
	sim.append(Simulation::CX(5, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(5, 21));
	sim.append(Simulation::CX(6, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(6, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(8, 21));
	sim.append(Simulation::CX(9, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(9, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(11, 21));
	sim.append(Simulation::CX(12, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(12, 21));
	sim.append(Simulation::CX(16, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(16, 21));
	sim.append(Simulation::CX(18, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(18, 21));
	sim.append(Simulation::CX(19, 21));
	sim.append(Simulation::U1(4.5683703, 21));
	sim.append(Simulation::CX(19, 21));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 19));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(1, 22));
	sim.append(Simulation::CX(3, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(3, 22));
	sim.append(Simulation::CX(4, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(4, 22));
	sim.append(Simulation::CX(5, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(5, 22));
	sim.append(Simulation::CX(6, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(6, 22));
	sim.append(Simulation::CX(14, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(14, 22));
	sim.append(Simulation::CX(15, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(15, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::U1(4.5683703, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(0, 23));
	sim.append(Simulation::CX(4, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(4, 23));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(7, 23));
	sim.append(Simulation::CX(10, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(10, 23));
	sim.append(Simulation::CX(13, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(13, 23));
	sim.append(Simulation::CX(15, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(15, 23));
	sim.append(Simulation::CX(17, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(17, 23));
	sim.append(Simulation::CX(18, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(18, 23));
	sim.append(Simulation::CX(20, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(20, 23));
	sim.append(Simulation::CX(21, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(21, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::U1(4.5683703, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 24));
	sim.append(Simulation::CX(0, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(0, 24));
	sim.append(Simulation::CX(2, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(2, 24));
	sim.append(Simulation::CX(3, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(3, 24));
	sim.append(Simulation::CX(5, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(5, 24));
	sim.append(Simulation::CX(6, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(6, 24));
	sim.append(Simulation::CX(7, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(7, 24));
	sim.append(Simulation::CX(10, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(10, 24));
	sim.append(Simulation::CX(14, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(14, 24));
	sim.append(Simulation::CX(15, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(15, 24));
	sim.append(Simulation::CX(16, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(16, 24));
	sim.append(Simulation::CX(17, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(17, 24));
	sim.append(Simulation::CX(20, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(20, 24));
	sim.append(Simulation::CX(22, 24));
	sim.append(Simulation::U1(4.5683703, 24));
	sim.append(Simulation::CX(22, 24));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 25));
	sim.append(Simulation::CX(2, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(2, 25));
	sim.append(Simulation::CX(4, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(4, 25));
	sim.append(Simulation::CX(5, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(5, 25));
	sim.append(Simulation::CX(8, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(8, 25));
	sim.append(Simulation::CX(10, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(10, 25));
	sim.append(Simulation::CX(13, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(13, 25));
	sim.append(Simulation::CX(15, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(15, 25));
	sim.append(Simulation::CX(16, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(16, 25));
	sim.append(Simulation::CX(17, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(17, 25));
	sim.append(Simulation::CX(20, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(20, 25));
	sim.append(Simulation::CX(21, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(21, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::U1(4.5683703, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 26));
	sim.append(Simulation::CX(0, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(0, 26));
	sim.append(Simulation::CX(1, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(1, 26));
	sim.append(Simulation::CX(10, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(10, 26));
	sim.append(Simulation::CX(11, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(11, 26));
	sim.append(Simulation::CX(12, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(12, 26));
	sim.append(Simulation::CX(14, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(14, 26));
	sim.append(Simulation::CX(15, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(15, 26));
	sim.append(Simulation::CX(16, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(16, 26));
	sim.append(Simulation::CX(17, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(17, 26));
	sim.append(Simulation::CX(18, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(18, 26));
	sim.append(Simulation::CX(21, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(21, 26));
	sim.append(Simulation::CX(24, 26));
	sim.append(Simulation::U1(4.5683703, 26));
	sim.append(Simulation::CX(24, 26));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 27));
	sim.append(Simulation::CX(0, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(0, 27));
	sim.append(Simulation::CX(3, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(3, 27));
	sim.append(Simulation::CX(4, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(4, 27));
	sim.append(Simulation::CX(6, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(6, 27));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 6));
	sim.append(Simulation::CX(8, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(8, 27));
	sim.append(Simulation::CX(9, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(9, 27));
	sim.append(Simulation::CX(11, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(11, 27));
	sim.append(Simulation::CX(15, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(15, 27));
	sim.append(Simulation::CX(16, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(16, 27));
	sim.append(Simulation::CX(17, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(17, 27));
	sim.append(Simulation::CX(18, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(18, 27));
	sim.append(Simulation::CX(23, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(23, 27));
	sim.append(Simulation::CX(24, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(24, 27));
	sim.append(Simulation::CX(25, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(25, 27));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::U1(4.5683703, 27));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 28));
	sim.append(Simulation::CX(0, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(0, 28));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 0));
	sim.append(Simulation::CX(1, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(1, 28));
	sim.append(Simulation::CX(2, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(2, 28));
	sim.append(Simulation::CX(4, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(4, 28));
	sim.append(Simulation::CX(5, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(5, 28));
	sim.append(Simulation::CX(8, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(8, 28));
	sim.append(Simulation::CX(11, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(11, 28));
	sim.append(Simulation::CX(12, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(12, 28));
	sim.append(Simulation::CX(15, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(15, 28));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 15));
	sim.append(Simulation::CX(18, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(18, 28));
	sim.append(Simulation::CX(25, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(25, 28));
	sim.append(Simulation::CX(26, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(26, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::U1(4.5683703, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 29));
	sim.append(Simulation::CX(1, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(1, 29));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 1));
	sim.append(Simulation::CX(2, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(2, 29));
	sim.append(Simulation::CX(7, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(7, 29));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 7));
	sim.append(Simulation::CX(9, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(9, 29));
	sim.append(Simulation::CX(10, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(10, 29));
	sim.append(Simulation::CX(13, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(13, 29));
	sim.append(Simulation::CX(17, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(17, 29));
	sim.append(Simulation::CX(21, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(21, 29));
	sim.append(Simulation::CX(24, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(24, 29));
	sim.append(Simulation::CX(26, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(26, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::U1(4.5683703, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 29));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 30));
	sim.append(Simulation::CX(2, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(2, 30));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 2));
	sim.append(Simulation::CX(3, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(3, 30));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 3));
	sim.append(Simulation::CX(8, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(8, 30));
	sim.append(Simulation::CX(16, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(16, 30));
	sim.append(Simulation::CX(17, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(17, 30));
	sim.append(Simulation::CX(20, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(20, 30));
	sim.append(Simulation::CX(21, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(21, 30));
	sim.append(Simulation::CX(22, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(22, 30));
	sim.append(Simulation::CX(24, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(24, 30));
	sim.append(Simulation::CX(25, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(25, 30));
	sim.append(Simulation::CX(26, 30));
	sim.append(Simulation::U1(4.5683703, 30));
	sim.append(Simulation::CX(26, 30));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 30));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 31));
	sim.append(Simulation::CX(4, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(4, 31));
	sim.append(Simulation::CX(5, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(5, 31));
	sim.append(Simulation::CX(8, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(8, 31));
	sim.append(Simulation::CX(11, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(11, 31));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 11));
	sim.append(Simulation::CX(12, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(12, 31));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 12));
	sim.append(Simulation::CX(13, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(13, 31));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 13));
	sim.append(Simulation::CX(16, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(16, 31));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 16));
	sim.append(Simulation::CX(18, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(18, 31));
	sim.append(Simulation::CX(20, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(20, 31));
	sim.append(Simulation::CX(22, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(22, 31));
	sim.append(Simulation::CX(23, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(23, 31));
	sim.append(Simulation::CX(24, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(24, 31));
	sim.append(Simulation::CX(26, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(26, 31));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 26));
	sim.append(Simulation::CX(27, 31));
	sim.append(Simulation::U1(4.5683703, 31));
	sim.append(Simulation::CX(27, 31));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 8));
	sim.append(Simulation::U3(1.5707963267948966, 0, 3.141592653589793, 32));
	sim.append(Simulation::CX(4, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(4, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 4));
	sim.append(Simulation::CX(5, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(5, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 5));
	sim.append(Simulation::CX(9, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(9, 32));
	sim.append(Simulation::CX(10, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(10, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 10));
	sim.append(Simulation::CX(14, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(14, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 14));
	sim.append(Simulation::CX(17, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(17, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 17));
	sim.append(Simulation::CX(18, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(18, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 18));
	sim.append(Simulation::CX(20, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(20, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 20));
	sim.append(Simulation::CX(21, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(21, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 21));
	sim.append(Simulation::CX(22, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(22, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 22));
	sim.append(Simulation::CX(23, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(23, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 23));
	sim.append(Simulation::CX(24, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(24, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 24));
	sim.append(Simulation::CX(25, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(25, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 25));
	sim.append(Simulation::CX(27, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(27, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 27));
	sim.append(Simulation::CX(28, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(28, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 28));
	sim.append(Simulation::CX(31, 32));
	sim.append(Simulation::U1(4.5683703, 32));
	sim.append(Simulation::CX(31, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 31));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 32));
	sim.append(Simulation::U3(5.3900259, -1.5707963267948966, 1.5707963267948966, 9));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=33;
	int n_gpus=2;
	int chunkSize=25;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}