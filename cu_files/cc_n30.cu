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
	sim.append(Simulation::CX(0, 29));
	sim.append(Simulation::CX(1, 29));
	sim.append(Simulation::CX(2, 29));
	sim.append(Simulation::CX(3, 29));
	sim.append(Simulation::CX(4, 29));
	sim.append(Simulation::CX(5, 29));
	sim.append(Simulation::CX(6, 29));
	sim.append(Simulation::CX(7, 29));
	sim.append(Simulation::CX(8, 29));
	sim.append(Simulation::CX(9, 29));
	sim.append(Simulation::CX(10, 29));
	sim.append(Simulation::CX(11, 29));
	sim.append(Simulation::CX(12, 29));
	sim.append(Simulation::CX(13, 29));
	sim.append(Simulation::CX(14, 29));
	sim.append(Simulation::CX(15, 29));
	sim.append(Simulation::CX(16, 29));
	sim.append(Simulation::CX(17, 29));
	sim.append(Simulation::CX(18, 29));
	sim.append(Simulation::CX(19, 29));
	sim.append(Simulation::CX(20, 29));
	sim.append(Simulation::CX(21, 29));
	sim.append(Simulation::CX(22, 29));
	sim.append(Simulation::CX(23, 29));
	sim.append(Simulation::CX(24, 29));
	sim.append(Simulation::CX(25, 29));
	sim.append(Simulation::CX(26, 29));
	sim.append(Simulation::CX(27, 29));
	sim.append(Simulation::CX(28, 29));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=30;
	int n_gpus=2;
	int chunkSize=25;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
