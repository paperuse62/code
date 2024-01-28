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
	sim.append(Simulation::CX(0, 33));
	sim.append(Simulation::CX(1, 33));
	sim.append(Simulation::CX(2, 33));
	sim.append(Simulation::CX(3, 33));
	sim.append(Simulation::CX(4, 33));
	sim.append(Simulation::CX(5, 33));
	sim.append(Simulation::CX(6, 33));
	sim.append(Simulation::CX(7, 33));
	sim.append(Simulation::CX(8, 33));
	sim.append(Simulation::CX(9, 33));
	sim.append(Simulation::CX(10, 33));
	sim.append(Simulation::CX(11, 33));
	sim.append(Simulation::CX(12, 33));
	sim.append(Simulation::CX(13, 33));
	sim.append(Simulation::CX(14, 33));
	sim.append(Simulation::CX(15, 33));
	sim.append(Simulation::CX(16, 33));
	sim.append(Simulation::CX(17, 33));
	sim.append(Simulation::CX(18, 33));
	sim.append(Simulation::CX(19, 33));
	sim.append(Simulation::CX(20, 33));
	sim.append(Simulation::CX(21, 33));
	sim.append(Simulation::CX(22, 33));
	sim.append(Simulation::CX(23, 33));
	sim.append(Simulation::CX(24, 33));
	sim.append(Simulation::CX(25, 33));
	sim.append(Simulation::CX(26, 33));
	sim.append(Simulation::CX(27, 33));
	sim.append(Simulation::CX(28, 33));
	sim.append(Simulation::CX(29, 33));
	sim.append(Simulation::CX(30, 33));
	sim.append(Simulation::CX(31, 33));
	sim.append(Simulation::CX(32, 33));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=34;
	int n_gpus=2;
	int chunkSize=25;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
