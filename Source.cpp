#include <stdio.h>
#include <math.h>
#include "mpi.h"

const int size = 1009;

int main(int argc, char** argv)
{
	int myid, np;
	int namelen;
	char proc_nam[MPI_MAX_PROCESSOR_NAME];


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Get_processor_name(proc_nam, &namelen);


	double mypi;
	int* masssize, *disp;
	double t1s, t2s, t1p, t2p;
	double* A, * B, *Cs, *Cp, *Aloc, *Cloc;
	MPI_Status st;

	masssize = new int[np];
	disp = new int[np];
	A = nullptr; B = nullptr; Cs = nullptr; Cp = nullptr; Aloc = nullptr; Cloc = nullptr;

	if (myid == 0)
	{
		A = new double[size * size];
		B = new double[size * size];
		Cs = new double[size * size];
		Cp = new double[size * size];

		printf("Size of matrix is %d\n", size);
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				A[i * size + j] = sin(i + j);
				B[i * size + j] = cos(i + j);
			}
	}
	else
	{
		A = nullptr;
		B = new double[size * size];
		Cs = nullptr;
		Cp = nullptr;
	}

	if (myid == 0)
	{
		t1s = MPI_Wtime();
		for (int i = 0; i < size; i++)
			for (int k = 0; k < size; k++)
				for (int j = 0; j < size; j++)
					Cs[i * size + j] = A[i * size + k] * B[k * size + j];

		t2s = MPI_Wtime();
		printf("without parallel C[n/2,n/2] = %f, time = %f\n", Cs[(size / 2) * size + size / 2], t2s - t1s);
	}

	t1p = MPI_Wtime();
	//find need size for each copy
	if (myid == 0)
	{
		for (int i = 0; i < np; i++)
			masssize[i] = (size / np) * size;
		for (int i = 0; i < size - (size / np) * np; i++)
			masssize[i] += size;
		disp[0] = 0;
		for (int i = 1; i < np; i++)
			disp[i] = disp[i - 1] + masssize[i - 1];
	}

	MPI_Bcast(masssize, np, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(disp, np, MPI_INT, 0, MPI_COMM_WORLD);

	//	printf("%d proc: %d size %d elem\n", myid, masssize[myid], disp[myid]);

	Aloc = new double[masssize[myid]];
	Cloc = new double[masssize[myid]];

	//send need information to process
	MPI_Scatterv(A, masssize, disp, MPI_DOUBLE, Aloc, masssize[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	//find part of C
	for (int i = 0; i < masssize[myid] / size; i++)
		for (int k = 0; k < size; k++)
			for (int j = 0; j < size; j++)
				Cloc[i * size + j] = Aloc[i * size + k] * B[k * size + j];

	//join all parts C in one martix
	MPI_Gatherv(Cloc, masssize[myid], MPI_DOUBLE, Cp, masssize, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	t2p = MPI_Wtime();

	if (myid == 0)
	{
		double max = 0;

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				if (max < fabs(Cp[i * size + j] - Cs[i * size + j]))
					max = fabs(Cp[i * size + j] - Cs[i * size + j]);
			}
		printf("with parallel C[n/2,n/2] = %f, time = %f\n", Cp[(size / 2) * size + size / 2], t2p - t1p);
		printf("Max error is %f\n", max);
		printf("without parallel VS parallel times = %f\n", (t2s - t1s) / (t2p - t1p));
	}

	MPI_Finalize();
	return 0;
}