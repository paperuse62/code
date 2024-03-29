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
	sim.append(Simulation::U1(1.9158955, 0));
	sim.append(Simulation::U1(0.90193458, 1));
	sim.append(Simulation::U1(1.3159483, 2));
	sim.append(Simulation::U1(0.77575935, 3));
	sim.append(Simulation::U1(1.0061371, 4));
	sim.append(Simulation::U1(0.72324323, 5));
	sim.append(Simulation::U1(2.51873, 6));
	sim.append(Simulation::U1(2.5197491, 7));
	sim.append(Simulation::U1(0.24787911, 8));
	sim.append(Simulation::U1(2.7127637, 9));
	sim.append(Simulation::U1(2.9664312, 10));
	sim.append(Simulation::U1(2.3432186, 11));
	sim.append(Simulation::U1(0.69702887, 12));
	sim.append(Simulation::U1(0.75610491, 13));
	sim.append(Simulation::U1(2.8125484, 14));
	sim.append(Simulation::U1(3.101376, 15));
	sim.append(Simulation::U1(1.1628422, 16));
	sim.append(Simulation::U1(1.5605353, 17));
	sim.append(Simulation::U1(1.3365387, 18));
	sim.append(Simulation::U1(1.3244761, 19));
	sim.append(Simulation::U1(0.91181244, 20));
	sim.append(Simulation::U1(0.71413017, 21));
	sim.append(Simulation::U1(1.3469317, 22));
	sim.append(Simulation::U1(0.22510665, 23));
	sim.append(Simulation::U1(2.80941, 24));
	sim.append(Simulation::U1(0.050821608, 25));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(2.7934282, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::RZ(1.2804871, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::RZ(1.621011, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::RZ(0.58413604, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::RZ(0.97748915, 5));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::RZ(2.2588826, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::RZ(2.8692917, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::RZ(1.3188165, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::RZ(2.7942117, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::RZ(0.48929622, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::RZ(2.7342733, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::RZ(0.61175857, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::RZ(2.9438995, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::RZ(2.6041232, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::RZ(1.1865173, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::RZ(2.5502053, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::RZ(1.4134301, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::RZ(1.3984205, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::RZ(1.7244872, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::RZ(2.18745, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::RZ(2.214484, 21));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::RZ(2.1277209, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::RZ(0.52438792, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::RZ(1.8780276, 24));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::RZ(1.5783824, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::RZ(0.3455544, 25));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::RZ(0.66611764, 24));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::RZ(2.460445, 23));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::RZ(0.44205969, 22));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::RZ(0.73643738, 21));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::RZ(0.54784404, 20));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::RZ(2.8906796, 19));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::RZ(2.6309005, 18));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::RZ(1.4973162, 17));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::RZ(2.6726049, 16));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::RZ(0.17248943, 15));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::RZ(0.58558452, 14));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::RZ(2.8603319, 13));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::RZ(1.6935914, 12));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::RZ(2.0836731, 11));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::RZ(1.8855929, 10));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::RZ(1.9596462, 9));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::RZ(2.5958827, 8));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::RZ(1.9849557, 7));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::RZ(0.32461377, 6));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::RZ(0.23667906, 5));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::RZ(2.6363604, 4));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::RZ(2.7862383, 3));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::RZ(0.94535421, 2));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(1.4950508, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RZ(1.0030982, 0));
	sim.append(Simulation::RZ(3.112718, 1));
	sim.append(Simulation::RZ(1.5561183, 2));
	sim.append(Simulation::RZ(1.4525627, 3));
	sim.append(Simulation::RZ(3.1319099, 4));
	sim.append(Simulation::RZ(0.51796279, 5));
	sim.append(Simulation::RZ(1.9566377, 6));
	sim.append(Simulation::RZ(2.5637, 7));
	sim.append(Simulation::RZ(1.3905149, 8));
	sim.append(Simulation::RZ(2.2118706, 9));
	sim.append(Simulation::RZ(0.67259752, 10));
	sim.append(Simulation::RZ(2.6312182, 11));
	sim.append(Simulation::RZ(2.2957384, 12));
	sim.append(Simulation::RZ(0.88166823, 13));
	sim.append(Simulation::RZ(1.924642, 14));
	sim.append(Simulation::RZ(0.54321013, 15));
	sim.append(Simulation::RZ(2.6973317, 16));
	sim.append(Simulation::RZ(2.6587892, 17));
	sim.append(Simulation::RZ(1.0886937, 18));
	sim.append(Simulation::RZ(2.3609919, 19));
	sim.append(Simulation::RZ(1.5127918, 20));
	sim.append(Simulation::RZ(0.31792107, 21));
	sim.append(Simulation::RZ(0.29767726, 22));
	sim.append(Simulation::RZ(1.746111, 23));
	sim.append(Simulation::RZ(1.2379431, 24));
	sim.append(Simulation::RZ(2.1063203, 25));
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
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=26;
	int n_gpus=2;
	int chunkSize=21;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
