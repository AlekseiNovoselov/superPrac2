#include <mpi.h>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <omp.h> //OpenMP

typedef std::vector<std::vector<double> > Matrix;

struct Neighboors {
	int up;
	int down;
	int left;
	int right;
};

struct Mesh_Step {
	std::vector<double> h1;
	std::vector<double> h2;
	std::vector<double> avrg_h1;
	std::vector<double> avrg_h2;
};

int Is_Power(int n)
{
    if (n <= 0){
        return(-1);
	}
	int i = 0;
    while(n % 2 == 0){
        i++;
        n >>= 1;
    }
    if ((n >> 1) == 0){
        return(i);
	} 
	else{
        return(-1);
	}
}

double f(double t, double q){
	return (pow(1 + t, q) - 1)/(pow(2, q) - 1);
}

double phi(double x, double y){
	return pow((1 - pow(x, 2)), 2) + pow((1 - pow(y, 2)), 2);
}

double F(double x, double y){
	return 4 * (2 - 3 * pow(x, 2) - 3 * pow(y, 2));
}

Matrix* Matrix_Plus(Matrix* a, Matrix* b) {
	Matrix* result(new Matrix());
	for (size_t i = 0; i < a->size(); i++) {
		result->push_back(std::vector<double>());
		for (size_t j = 0; j < (*a)[0].size(); ++j) {
			(*result)[i].push_back((*a)[i][j] + (*b)[i][j]);
		}
	}
	return result;
}

Matrix* Matrix_Prod_Scalar(Matrix* a, double alpha) {
	Matrix* result(new Matrix());
	for (size_t i = 0; i < a->size(); i++) {
		result->push_back(std::vector<double>());
		for (size_t j = 0; j < (*a)[0].size(); ++j) {
			(*result)[i].push_back((*a)[i][j] * alpha);
		}
	}
	return result;
}

double Dot (Matrix* a, Matrix* b, const Mesh_Step& stp){
	double result = 0;
	#pragma omp parallel for reduction(+:result)
	for(int i = 1; i < a->size() - 1; i++){
		for(int j = 1; j < (*a)[0].size() - 1; j++){
			result += stp.avrg_h1[j] * stp.avrg_h2[i] * (*a)[i][j] * (*b)[i][j];
		}
	}
	return result;
}

Matrix* Laplacian (Matrix* a, const Mesh_Step& stp){
	Matrix* result(new Matrix(a->size(), std::vector<double>((*a)[0].size(), 0)));
	#pragma omp parallel for
	for(int i = 1; i < a->size() - 1; i++){
		for(int j = 1; j < (*a)[0].size() - 1; j++){
		(*result)[i][j] = 1 / stp.avrg_h1[j] * (((*a)[i][j] - (*a)[i][j - 1]) / stp.h1[j - 1] - ((*a)[i][j + 1] - (*a)[i][j]) / stp.h1[j]) + 
		                  1 / stp.avrg_h2[i] * (((*a)[i][j] - (*a)[i - 1][j]) / stp.h2[i - 1] - ((*a)[i + 1][j] - (*a)[i][j]) / stp.h2[i]);
		}
	}
	return result;
}

double Count_Coef(int pid, int np, std::vector<double> a){
	MPI_Status *s = new MPI_Status;
	double coef = 0;
	if (pid == 0){
		for (int i = 1; i < np; i++){
			std::vector<double> tmp (2, 0);
			MPI_Recv(&tmp[0], 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, s);
			a[0] += tmp[0];
			a[1] += tmp[1];
		}
		coef = a[0]/a[1];
		for (int i = 1; i < np; i++){
			MPI_Send(&coef, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
	else{
		MPI_Send(&a[0], 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&coef, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, s);
	}
	return coef;
	delete s;
}

void MPI_Send_Matrix(Matrix* a, const Neighboors& nb){
	MPI_Status *s = new MPI_Status;
	size_t clmns = a->size();
	size_t rws = (*a)[0].size();
	if (nb.up != -1){
		MPI_Sendrecv(&((*a)[1][0]), rws, MPI_DOUBLE, nb.up, 0, &((*a)[0][0]), rws, MPI_DOUBLE, nb.up, 0, MPI_COMM_WORLD, s);
		/*MPI_Send(&((*a)[1][0]), rws, MPI_DOUBLE, nb.up, 0, MPI_COMM_WORLD);
		MPI_Recv(&((*a)[0][0]), rws, MPI_DOUBLE, nb.up, 0, MPI_COMM_WORLD, s);*/
	}
	if (nb.down != -1){
		MPI_Sendrecv(&((*a)[clmns - 2][0]), rws, MPI_DOUBLE, nb.down, 0, &((*a)[clmns - 1][0]), rws, MPI_DOUBLE, nb.down, 0, MPI_COMM_WORLD, s);
		/*MPI_Send(&((*a)[clmns - 2][0]), rws, MPI_DOUBLE, nb.down, 0, MPI_COMM_WORLD);
		MPI_Recv(&((*a)[clmns - 1][0]), rws, MPI_DOUBLE, nb.down, 0, MPI_COMM_WORLD, s);*/
	}
	if (nb.left != -1){
		std::vector<double> tmp_send(clmns, 0);
		std::vector<double> tmp_recv(clmns, 0);
		for (size_t i = 0; i < clmns; i++){
			tmp_send[i] = (*a)[i][1];
		}
		MPI_Sendrecv(&tmp_send[0], clmns, MPI_DOUBLE, nb.left, 0, &tmp_recv[0], clmns, MPI_DOUBLE, nb.left, 0, MPI_COMM_WORLD, s);
		/*MPI_Send(&tmp[0], clmns, MPI_DOUBLE, nb.left, 0, MPI_COMM_WORLD);
		MPI_Recv(&tmp[0], clmns, MPI_DOUBLE, nb.left, 0, MPI_COMM_WORLD, s);*/
		for (size_t i = 0; i < clmns; i++){
			(*a)[i][0] = tmp_recv[i];
		}
	}
	if (nb.right != -1){
		std::vector<double> tmp_send(clmns, 0);
		std::vector<double> tmp_recv(clmns, 0);
		for (size_t i = 0; i < clmns; i++){
			tmp_send[i] = (*a)[i][rws - 2];
		}
		MPI_Sendrecv(&tmp_send[0], clmns, MPI_DOUBLE, nb.right, 0, &tmp_recv[0], clmns, MPI_DOUBLE, nb.right, 0, MPI_COMM_WORLD, s);
		/*MPI_Send(&tmp[0], clmns, MPI_DOUBLE, nb.right, 0, MPI_COMM_WORLD);
		MPI_Recv(&tmp[0], clmns, MPI_DOUBLE, nb.right, 0, MPI_COMM_WORLD, s);*/
		for (size_t i = 0; i < clmns; i++){
			(*a)[i][rws - 1] = tmp_recv[i];
		}
	}
	delete s;
}

void Matrix_Out(Matrix* a){
	for(size_t i = 0; i < a->size(); i++){
		for(size_t j = 0; j < (*a)[i].size(); j++){
			std::cout << (*a)[i][j] << " ";
		}
		std::cout << "\n";
	}
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
	int pid, np;
	double strt_time = MPI_Wtime();
	MPI_Status *s = new MPI_Status;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	int pwr = Is_Power(np);
	if (pwr < 0){
		if (pid == 0){
			std::cout << "The number of procs must be a power of 2.\n";
		}
		delete s;
		MPI_Finalize();
		return (1);
	}
	if (argc != 2){
		if (pid == 0){
			std::cout << "Wrong parameter set.\n";
		}
		delete s;
		MPI_Finalize();
		return(2);
	}
	size_t n0_n1 = atoi(argv[1]) + 1;
	if (n0_n1 <= 0){
		if (pid == 0){
			std::cout << "Mesh size must be positive.\n";
		}
		delete s;
		MPI_Finalize();
		return (3);
	}
	const double q = 1.5;
	const double a1 = 0;
	const double a2 = 1;
	const double b1 = 0;
	const double b2 = 1;
	const double eps = 1e-4;
	size_t p_in_rw = 0;
	(pwr % 2 == 0) ? p_in_rw = pow(2, pwr / 2) : p_in_rw = pow(2, (pwr + 1) / 2);
	size_t div = n0_n1 / p_in_rw;
	size_t mod = n0_n1 % p_in_rw;
	size_t tmp_id = pid % p_in_rw;
	size_t p_rws_el = div + (tmp_id < mod ? 1 : 0);
	size_t strt_rw_el = 0;
	if (tmp_id < mod){
		strt_rw_el = (div + 1) * tmp_id;
	}
	else{
		strt_rw_el = (div + 1) * mod + div * (tmp_id - mod);
	}
	size_t p_in_clmn = np / p_in_rw;
	div = n0_n1 / p_in_clmn;
	mod = n0_n1 % p_in_clmn;
	tmp_id = pid / p_in_rw;
	size_t p_clmns_el = div + (tmp_id < mod ? 1 : 0);
	size_t strt_clmn_el = 0;
	if (tmp_id < mod){
		strt_clmn_el = (div + 1) * tmp_id;
	}
	else{
		strt_clmn_el = (div + 1) * mod + div * (tmp_id - mod);
	}
	Neighboors p_nghbrs = {-1, -1, -1, -1};
	if (pid / p_in_rw != 0){
		p_nghbrs.up = pid - p_in_rw;
		strt_clmn_el--;
		p_clmns_el++;
	}
	if (pid / p_in_rw != p_in_clmn - 1){
		p_nghbrs.down = pid + p_in_rw;
		p_clmns_el++;
	}
	if (pid % p_in_rw != 0){
		p_nghbrs.left = pid - 1;
		strt_rw_el--;
		p_rws_el++;
	}
	if ((pid + 1) % p_in_rw != 0){
		p_nghbrs.right = pid + 1;
		p_rws_el++;
	}
	/*std::cout << "p_clmns_el = " << p_clmns_el << "\n";
	std::cout << "p_rws_el = " << p_rws_el	<< "\n";
	std::cout << "p_in_clmn = " << p_in_clmn << "\n";
	std::cout << "p_in_rw = " << p_in_rw << "\n";
	std::cout << "strt_clmn_el = " << strt_clmn_el << "\n";
	std::cout << "strt_rw_el = " << strt_rw_el << "\n";*/
	std::vector<double> x(p_rws_el, 0);
	Mesh_Step h;
	h.h1.resize(p_rws_el, 0); //h[i] = x[i + 1] - x[i]
	h.avrg_h1.resize(p_rws_el, 0); //avh[i] = 0.5( h[i] + h[i - 1] ) = 0.5( x[i + 1] - x[i] + x[i] - x[i - 1] )
	for(size_t i = 0; i < p_rws_el; i++){
			x[i] = a2 * f(static_cast<double>((strt_rw_el + i)) / (n0_n1 - 1), q) + a1 * (1 - f(static_cast<double>((strt_rw_el + i)) / (n0_n1 - 1), q));
	}
	h.h1[0] = x[1] - x[0];
	for(size_t i = 1; i < p_rws_el - 1; i++){
		h.h1[i] = x[i + 1] - x[i];
		h.avrg_h1[i] = 0.5 * (x[i + 1] - x[i - 1]);
	}
	std::vector<double> y(p_clmns_el, 0);
	h.h2.resize(p_clmns_el + 1, 0);
	h.avrg_h2.resize(p_clmns_el, 0); 
	for (size_t i = 0; i < p_clmns_el; i++){
		y[i] = b2 * f(static_cast<double>((strt_clmn_el + i)) / (n0_n1 - 1), q) + b1 * (1 - f(static_cast<double>((strt_clmn_el + i)) / (n0_n1 - 1), q));/
	}
	h.h2[0] = x[1] - x[0];
	for(size_t i = 1; i < p_clmns_el - 1; i++){
		h.h2[i] = y[i + 1] - y[i];
		h.avrg_h2[i] = 0.5 * (y[i + 1] - y[i - 1]);
	}
	/*for (size_t i = 0; i < x.size(); i++){
		std::cout << "x[i] = " << x[i] << " ";
	}
	std::cout << "\n";
	for (size_t i = 0; i < y.size(); i++){
		std::cout << "y[i] = " << y[i] << " ";
	}
	std::cout << "\n";
	for (size_t i = 0; i < h.h1.size(); i++){
		std::cout << "h1[i] = " << h.h1[i] << " ";
	}
	std::cout << "\n";
	for (size_t i = 0; i < h.h2.size(); i++){
		std::cout << "h2[i] = " << h.h2[i] << " ";
	}
	std::cout << "\n";
	for (size_t i = 0; i < h.avrg_h1.size(); i++){
		std::cout << "avrg_h1[i] = " << h.avrg_h1[i] << " ";
	}
	std::cout << "\n";
	for (size_t i = 0; i < h.avrg_h2.size(); i++){
		std::cout << "avrg_h2[i] = " << h.avrg_h2[i] << " ";
	}
	std::cout << "\n";*/
	Matrix* mtrx(new Matrix());
	Matrix* r_mtrx(new Matrix(p_clmns_el, std::vector<double>(p_rws_el, 0)));
	Matrix* g_mtrx(new Matrix(p_clmns_el, std::vector<double>(p_rws_el, 0)));
	for(size_t i = 0; i < p_clmns_el; i++){
		mtrx->push_back(std::vector<double>());
		for(size_t j = 0; j < p_rws_el; j++){
			if ((p_nghbrs.up == -1) && (i == 0)){
				(*mtrx)[i].push_back(phi(x[j],y[i]));
			}
			else if ((p_nghbrs.down == -1) && (i == (p_clmns_el - 1))){
				(*mtrx)[i].push_back(phi(x[j], y[i]));
			}
			else if ((p_nghbrs.left == -1) && (j == 0)){
				(*mtrx)[i].push_back(phi(x[j], y[i]));
			}
			else if ((p_nghbrs.right == -1) && (j == (p_rws_el - 1))){
				(*mtrx)[i].push_back(phi(x[j], y[i]));
			}
			else{
				(*mtrx)[i].push_back(1);
			}
		}
	}
	//std::cout << "MMATRIX\n";
	//Matrix_Out(mtrx);
	size_t itr = -1;
	double nrm = 0;
	do{
		++itr;
		double alpha = 0;
		double tau = 0;
		MPI_Send_Matrix(mtrx, p_nghbrs);
		Matrix* lplc = Laplacian(mtrx, h);
		for(size_t i = 1; i < p_clmns_el - 1; i++){
			for(size_t j = 1; j < p_rws_el - 1; j++){
				(*r_mtrx)[i][j] = (*lplc)[i][j]	- F(x[j], y[i]);
			}
		}
		delete lplc;
		/*if (itr == 0){
			std::cout << "RMATRIX\n";
			Matrix_Out(r_mtrx);
		}*/
		Matrix* tmp_mtrx1;
		Matrix* tmp_mtrx2;
		MPI_Send_Matrix(r_mtrx,p_nghbrs);
		std::vector<double> rggg(2, 0);
		if (itr != 0){
			lplc = Laplacian(r_mtrx, h);
			rggg[0] = Dot(lplc, g_mtrx, h);
			delete lplc;
			lplc = Laplacian(g_mtrx, h);
			rggg[1] = Dot(g_mtrx, lplc, h);
			delete lplc;
			alpha = Count_Coef(pid, np, rggg);
		}
		tmp_mtrx1 = Matrix_Prod_Scalar(g_mtrx, -alpha);
		delete g_mtrx;
		g_mtrx = Matrix_Plus(r_mtrx, tmp_mtrx1);
		delete tmp_mtrx1;
		MPI_Send_Matrix(g_mtrx,p_nghbrs);
		rggg[0] = Dot(r_mtrx, g_mtrx, h);
		lplc = Laplacian(g_mtrx, h);
		rggg[1] = Dot(g_mtrx, lplc, h);
		delete lplc;
		tau = Count_Coef(pid, np, rggg);
		tmp_mtrx1 = Matrix_Prod_Scalar(g_mtrx, -tau);
		nrm = Dot(tmp_mtrx1, tmp_mtrx1, h);
		tmp_mtrx2 = Matrix_Plus(mtrx, tmp_mtrx1);
		delete mtrx;
		mtrx = tmp_mtrx2;
		delete tmp_mtrx1;
		if (pid == 0){
			for (int i = 1; i < np; i++){
				double tmp = 0;
				MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, s);
				nrm += tmp;
			}
			nrm = sqrt(nrm);
			for (int i = 1; i < np; i++){
				MPI_Send(&nrm, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
		}
		else{
			MPI_Send(&nrm, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Recv(&nrm, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, s);
		}
		//std::cout << "Debug 1: nrm = " << nrm << "\n";
	}
    while(nrm > eps);
	delete g_mtrx;
	delete r_mtrx;
	Matrix* err_mtrx(new Matrix(p_clmns_el, std::vector<double>(p_rws_el, 0)));
	for(size_t i = 1; i < p_clmns_el - 1; i++){
		for(size_t j = 1; j < p_rws_el - 1; j++){
			(*err_mtrx)[i][j] = (*mtrx)[i][j] - phi(x[j], y[i]);
		}
	}
	double err = Dot(err_mtrx, err_mtrx, h);
	if (pid == 0){
		for (int i = 1; i < np; i++){
			double tmp = 0;
			MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, s);
			err += tmp;
		}
		err = sqrt(err);
	}
	else{
		MPI_Send(&err, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	delete err_mtrx;
	if (pid == 0){
		std::cout << "Success! Time = " << MPI_Wtime() - strt_time << "\n";
		std::cout << "Number of iteration = " << itr << "\n";
		std::cout << "Error = " << err << "\n";
	}
	delete mtrx;
	delete s;
	/*for (int i = 0; i < np; i++){
		if (pid == i){
			std::cout << pid << ":\n";
			Matrix_Out(mtrx);
		}
	}*/
	MPI_Finalize();
	return 0;
}