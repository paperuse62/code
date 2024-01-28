#include <stdio.h>
#include "bmlqsim.cuh"
//Use the SVSim namespace to enable C++/CUDA APIs
using namespace SVSim;

void prepare_circuit(Simulation &sim)
{
	sim.append(Simulation::H(0));
	sim.append(Simulation::H(1));
	sim.append(Simulation::H(2));
	sim.append(Simulation::H(3));
	sim.append(Simulation::H(4));
	sim.append(Simulation::H(5));
	sim.append(Simulation::H(6));
	sim.append(Simulation::H(7));
	sim.append(Simulation::H(8));
	sim.append(Simulation::H(9));
	sim.append(Simulation::H(10));
	sim.append(Simulation::H(11));
	sim.append(Simulation::H(12));
	sim.append(Simulation::H(13));
	sim.append(Simulation::H(14));
	sim.append(Simulation::H(15));
	sim.append(Simulation::H(16));
	sim.append(Simulation::H(17));
	sim.append(Simulation::H(18));
	sim.append(Simulation::H(19));
	sim.append(Simulation::H(20));
	sim.append(Simulation::H(21));
	sim.append(Simulation::H(22));
	sim.append(Simulation::H(23));
	sim.append(Simulation::H(24));
	sim.append(Simulation::H(25));
	sim.append(Simulation::H(26));
	sim.append(Simulation::H(27));
	sim.append(Simulation::H(28));
	sim.append(Simulation::H(29));
	sim.append(Simulation::H(30));
	sim.append(Simulation::H(31));
	sim.append(Simulation::H(32));
	sim.append(Simulation::RZ(0.8355633, 0));
	sim.append(Simulation::RZ(-0.8355633, 1));
	sim.append(Simulation::RZ(-0.8355633, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(0.8355633, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(-0.78222362, 2));
	sim.append(Simulation::RZ(0.78222362, 3));
	sim.append(Simulation::RZ(0.78222362, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::RZ(-0.78222362, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::RZ(-1.0057915, 4));
	sim.append(Simulation::RZ(1.0057915, 5));
	sim.append(Simulation::RZ(1.0057915, 5));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::RZ(-1.0057915, 5));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::RZ(-1.2194914, 6));
	sim.append(Simulation::RZ(1.2194914, 7));
	sim.append(Simulation::RZ(1.2194914, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::RZ(-1.2194914, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::RZ(1.4719388, 8));
	sim.append(Simulation::RZ(-1.4719388, 9));
	sim.append(Simulation::RZ(-1.4719388, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::RZ(1.4719388, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::RZ(-0.92246519, 10));
	sim.append(Simulation::RZ(0.92246519, 11));
	sim.append(Simulation::RZ(0.92246519, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::RZ(-0.92246519, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::RZ(-1.5420291, 12));
	sim.append(Simulation::RZ(1.5420291, 13));
	sim.append(Simulation::RZ(1.5420291, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::RZ(-1.5420291, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::RZ(0.12770177, 14));
	sim.append(Simulation::RZ(-0.12770177, 15));
	sim.append(Simulation::RZ(-0.12770177, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::RZ(0.12770177, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::RZ(0.67391245, 16));
	sim.append(Simulation::RZ(-0.67391245, 17));
	sim.append(Simulation::RZ(-0.67391245, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::RZ(0.67391245, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::RZ(1.0798858, 18));
	sim.append(Simulation::RZ(-1.0798858, 19));
	sim.append(Simulation::RZ(-1.0798858, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::RZ(1.0798858, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::RZ(1.7712903, 20));
	sim.append(Simulation::RZ(-1.7712903, 21));
	sim.append(Simulation::RZ(-1.7712903, 21));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::RZ(1.7712903, 21));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::RZ(0.42500555, 22));
	sim.append(Simulation::RZ(-0.42500555, 23));
	sim.append(Simulation::RZ(-0.42500555, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::RZ(0.42500555, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::RZ(0.75930512, 24));
	sim.append(Simulation::RZ(-0.75930512, 25));
	sim.append(Simulation::RZ(-0.75930512, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::RZ(0.75930512, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::RZ(0.56391453, 26));
	sim.append(Simulation::RZ(-0.56391453, 27));
	sim.append(Simulation::RZ(-0.56391453, 27));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::RZ(0.56391453, 27));
	sim.append(Simulation::CX(26, 27));
	sim.append(Simulation::RZ(0.36413661, 28));
	sim.append(Simulation::RZ(-0.36413661, 29));
	sim.append(Simulation::RZ(-0.36413661, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::RZ(0.36413661, 29));
	sim.append(Simulation::CX(28, 29));
	sim.append(Simulation::RZ(1.5011633, 30));
	sim.append(Simulation::RZ(-1.5011633, 31));
	sim.append(Simulation::RZ(-1.5011633, 31));
	sim.append(Simulation::CX(30, 31));
	sim.append(Simulation::RZ(1.5011633, 31));
	sim.append(Simulation::CX(30, 31));
	sim.append(Simulation::RZ(1.6751132, 1));
	sim.append(Simulation::RZ(-1.6751132, 2));
	sim.append(Simulation::RZ(-1.6751132, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::RZ(1.6751132, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::RZ(1.0373497, 3));
	sim.append(Simulation::RZ(-1.0373497, 4));
	sim.append(Simulation::RZ(-1.0373497, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::RZ(1.0373497, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::RZ(1.3044758, 5));
	sim.append(Simulation::RZ(-1.3044758, 6));
	sim.append(Simulation::RZ(-1.3044758, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::RZ(1.3044758, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::RZ(0.7413099, 7));
	sim.append(Simulation::RZ(-0.7413099, 8));
	sim.append(Simulation::RZ(-0.7413099, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::RZ(0.7413099, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::RZ(0.8630933, 9));
	sim.append(Simulation::RZ(-0.8630933, 10));
	sim.append(Simulation::RZ(-0.8630933, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::RZ(0.8630933, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::RZ(-1.1169214, 11));
	sim.append(Simulation::RZ(1.1169214, 12));
	sim.append(Simulation::RZ(1.1169214, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::RZ(-1.1169214, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::RZ(-0.91111811, 13));
	sim.append(Simulation::RZ(0.91111811, 14));
	sim.append(Simulation::RZ(0.91111811, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::RZ(-0.91111811, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::RZ(-1.1587232, 15));
	sim.append(Simulation::RZ(1.1587232, 16));
	sim.append(Simulation::RZ(1.1587232, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::RZ(-1.1587232, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::RZ(0.72339072, 17));
	sim.append(Simulation::RZ(-0.72339072, 18));
	sim.append(Simulation::RZ(-0.72339072, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::RZ(0.72339072, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::RZ(1.9546081, 19));
	sim.append(Simulation::RZ(-1.9546081, 20));
	sim.append(Simulation::RZ(-1.9546081, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::RZ(1.9546081, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::RZ(-0.36618039, 21));
	sim.append(Simulation::RZ(0.36618039, 22));
	sim.append(Simulation::RZ(0.36618039, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::RZ(-0.36618039, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::RZ(1.8754904, 23));
	sim.append(Simulation::RZ(-1.8754904, 24));
	sim.append(Simulation::RZ(-1.8754904, 24));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::RZ(1.8754904, 24));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::RZ(-0.011943041, 25));
	sim.append(Simulation::RZ(0.011943041, 26));
	sim.append(Simulation::RZ(0.011943041, 26));
	sim.append(Simulation::CX(25, 26));
	sim.append(Simulation::RZ(-0.011943041, 26));
	sim.append(Simulation::CX(25, 26));
	sim.append(Simulation::RZ(0.12756422, 27));
	sim.append(Simulation::RZ(-0.12756422, 28));
	sim.append(Simulation::RZ(-0.12756422, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::RZ(0.12756422, 28));
	sim.append(Simulation::CX(27, 28));
	sim.append(Simulation::RZ(-0.88391875, 29));
	sim.append(Simulation::RZ(0.88391875, 30));
	sim.append(Simulation::RZ(0.88391875, 30));
	sim.append(Simulation::CX(29, 30));
	sim.append(Simulation::RZ(-0.88391875, 30));
	sim.append(Simulation::CX(29, 30));
	sim.append(Simulation::RZ(0.80376835, 31));
	sim.append(Simulation::RZ(-0.80376835, 32));
	sim.append(Simulation::RZ(-0.80376835, 32));
	sim.append(Simulation::CX(31, 32));
	sim.append(Simulation::RZ(0.80376835, 32));
	sim.append(Simulation::CX(31, 32));
	sim.append(Simulation::H(0));
	sim.append(Simulation::RZ(0, 0));
	sim.append(Simulation::H(0));
	sim.append(Simulation::RZ(0, 0));
	sim.append(Simulation::H(1));
	sim.append(Simulation::RZ(0, 1));
	sim.append(Simulation::H(1));
	sim.append(Simulation::RZ(0, 1));
	sim.append(Simulation::H(2));
	sim.append(Simulation::RZ(0, 2));
	sim.append(Simulation::H(2));
	sim.append(Simulation::RZ(0, 2));
	sim.append(Simulation::H(3));
	sim.append(Simulation::RZ(0, 3));
	sim.append(Simulation::H(3));
	sim.append(Simulation::RZ(0, 3));
	sim.append(Simulation::H(4));
	sim.append(Simulation::RZ(0, 4));
	sim.append(Simulation::H(4));
	sim.append(Simulation::RZ(0, 4));
	sim.append(Simulation::H(5));
	sim.append(Simulation::RZ(0, 5));
	sim.append(Simulation::H(5));
	sim.append(Simulation::RZ(0, 5));
	sim.append(Simulation::H(6));
	sim.append(Simulation::RZ(0, 6));
	sim.append(Simulation::H(6));
	sim.append(Simulation::RZ(0, 6));
	sim.append(Simulation::H(7));
	sim.append(Simulation::RZ(0, 7));
	sim.append(Simulation::H(7));
	sim.append(Simulation::RZ(0, 7));
	sim.append(Simulation::H(8));
	sim.append(Simulation::RZ(0, 8));
	sim.append(Simulation::H(8));
	sim.append(Simulation::RZ(0, 8));
	sim.append(Simulation::H(9));
	sim.append(Simulation::RZ(0, 9));
	sim.append(Simulation::H(9));
	sim.append(Simulation::RZ(0, 9));
	sim.append(Simulation::H(10));
	sim.append(Simulation::RZ(0, 10));
	sim.append(Simulation::H(10));
	sim.append(Simulation::RZ(0, 10));
	sim.append(Simulation::H(11));
	sim.append(Simulation::RZ(0, 11));
	sim.append(Simulation::H(11));
	sim.append(Simulation::RZ(0, 11));
	sim.append(Simulation::H(12));
	sim.append(Simulation::RZ(0, 12));
	sim.append(Simulation::H(12));
	sim.append(Simulation::RZ(0, 12));
	sim.append(Simulation::H(13));
	sim.append(Simulation::RZ(0, 13));
	sim.append(Simulation::H(13));
	sim.append(Simulation::RZ(0, 13));
	sim.append(Simulation::H(14));
	sim.append(Simulation::RZ(0, 14));
	sim.append(Simulation::H(14));
	sim.append(Simulation::RZ(0, 14));
	sim.append(Simulation::H(15));
	sim.append(Simulation::RZ(0, 15));
	sim.append(Simulation::H(15));
	sim.append(Simulation::RZ(0, 15));
	sim.append(Simulation::H(16));
	sim.append(Simulation::RZ(0, 16));
	sim.append(Simulation::H(16));
	sim.append(Simulation::RZ(0, 16));
	sim.append(Simulation::H(17));
	sim.append(Simulation::RZ(0, 17));
	sim.append(Simulation::H(17));
	sim.append(Simulation::RZ(0, 17));
	sim.append(Simulation::H(18));
	sim.append(Simulation::RZ(0, 18));
	sim.append(Simulation::H(18));
	sim.append(Simulation::RZ(0, 18));
	sim.append(Simulation::H(19));
	sim.append(Simulation::RZ(0, 19));
	sim.append(Simulation::H(19));
	sim.append(Simulation::RZ(0, 19));
	sim.append(Simulation::H(20));
	sim.append(Simulation::RZ(0, 20));
	sim.append(Simulation::H(20));
	sim.append(Simulation::RZ(0, 20));
	sim.append(Simulation::H(21));
	sim.append(Simulation::RZ(0, 21));
	sim.append(Simulation::H(21));
	sim.append(Simulation::RZ(0, 21));
	sim.append(Simulation::H(22));
	sim.append(Simulation::RZ(0, 22));
	sim.append(Simulation::H(22));
	sim.append(Simulation::RZ(0, 22));
	sim.append(Simulation::H(23));
	sim.append(Simulation::RZ(0, 23));
	sim.append(Simulation::H(23));
	sim.append(Simulation::RZ(0, 23));
	sim.append(Simulation::H(24));
	sim.append(Simulation::RZ(0, 24));
	sim.append(Simulation::H(24));
	sim.append(Simulation::RZ(0, 24));
	sim.append(Simulation::H(25));
	sim.append(Simulation::RZ(0, 25));
	sim.append(Simulation::H(25));
	sim.append(Simulation::RZ(0, 25));
	sim.append(Simulation::H(26));
	sim.append(Simulation::RZ(0, 26));
	sim.append(Simulation::H(26));
	sim.append(Simulation::RZ(0, 26));
	sim.append(Simulation::H(27));
	sim.append(Simulation::RZ(0, 27));
	sim.append(Simulation::H(27));
	sim.append(Simulation::RZ(0, 27));
	sim.append(Simulation::H(28));
	sim.append(Simulation::RZ(0, 28));
	sim.append(Simulation::H(28));
	sim.append(Simulation::RZ(0, 28));
	sim.append(Simulation::H(29));
	sim.append(Simulation::RZ(0, 29));
	sim.append(Simulation::H(29));
	sim.append(Simulation::RZ(0, 29));
	sim.append(Simulation::H(30));
	sim.append(Simulation::RZ(0, 30));
	sim.append(Simulation::H(30));
	sim.append(Simulation::RZ(0, 30));
	sim.append(Simulation::H(31));
	sim.append(Simulation::RZ(0, 31));
	sim.append(Simulation::H(31));
	sim.append(Simulation::RZ(0, 31));
	sim.append(Simulation::H(32));
	sim.append(Simulation::RZ(0, 32));
	sim.append(Simulation::H(32));
	sim.append(Simulation::RZ(0, 32));
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