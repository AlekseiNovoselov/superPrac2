#include <mpi.h>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <omp.h> //OpenMP
using namespace std;

typedef std::vector<std::vector<double> > Matrix;

// VAR 8
const double q = 2.0/3.0;
const double A1 = 0;
const double A2 = 2;
const double B1 = 0;
const double B2 = 2;
const double EPS = 1e-4;

struct Neighbors {
    int up;
    int down;
    int left;
    int right;
};

struct GridStep {
    std::vector<double> h1;
    std::vector<double> h2;
    std::vector<double> avg_h1;
    std::vector<double> avg_h2;
};

// Функция для вычисления значения для нелинейной сетки
double f(double t, double q){
    return (pow(1 + t, q) - 1)/(pow(2, q) - 1);
}

// Функция граничного условия
double phi(double x, double y){
    return 1+sin(x*y);
}

// Функция градиента
double F(double x, double y){
    return (x*x+y*y)*sin(x*y);
}

Matrix* sumMatrix(Matrix *a, Matrix *b) {
    Matrix* result(new Matrix());
    for (size_t i = 0; i < a->size(); i++) {
        result->push_back(std::vector<double>());
        for (size_t j = 0; j < (*a)[0].size(); ++j) {
            (*result)[i].push_back((*a)[i][j] + (*b)[i][j]);
        }
    }
    return result;
}

Matrix* scaleMatrix(Matrix *a, double alpha) {
    Matrix* result(new Matrix());
    for (size_t i = 0; i < a->size(); i++) {
        result->push_back(std::vector<double>());
        for (size_t j = 0; j < (*a)[0].size(); ++j) {
            (*result)[i].push_back((*a)[i][j] * alpha);
        }
    }
    return result;
}

double scalarMatrixProduct(Matrix *a, Matrix *b, const GridStep &stp){
    double result = 0;
#pragma omp parallel for reduction(+:result)
    for(int i = 1; i < a->size() - 1; i++){
        for(int j = 1; j < (*a)[0].size() - 1; j++){
            result += stp.avg_h1[j] * stp.avg_h2[i] * (*a)[i][j] * (*b)[i][j];
        }
    }
    return result;
}

Matrix* Laplacian (Matrix* a, const GridStep& stp){
    Matrix* result(new Matrix(a->size(), std::vector<double>((*a)[0].size(), 0)));
#pragma omp parallel for
    for(int i = 1; i < a->size() - 1; i++){
        for(int j = 1; j < (*a)[0].size() - 1; j++){
            (*result)[i][j] = 1 / stp.avg_h1[j] * (((*a)[i][j] - (*a)[i][j - 1]) / stp.h1[j - 1] - ((*a)[i][j + 1] - (*a)[i][j]) / stp.h1[j]) +
                              1 / stp.avg_h2[i] * (((*a)[i][j] - (*a)[i - 1][j]) / stp.h2[i - 1] - ((*a)[i + 1][j] - (*a)[i][j]) / stp.h2[i]);
        }
    }
    return result;
}

double countAlpha(int pid, int np, std::vector<double> a){
    MPI_Status *s = new MPI_Status;
    double alpha = 0;
    if (pid == 0){
        for (int i = 1; i < np; i++){
            std::vector<double> tmp (2, 0);
            MPI_Recv(&tmp[0], 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, s);
            a[0] += tmp[0];
            a[1] += tmp[1];
        }
        alpha = a[0]/a[1];
        for (int i = 1; i < np; i++){
            MPI_Send(&alpha, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Send(&a[0], 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&alpha, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, s);
    }
    return alpha;
    delete s;
}

void sendMatrixMPI(Matrix *matrix, const Neighbors &nb){
    MPI_Status *status = new MPI_Status;
    size_t columns = matrix->size();
    size_t rows = (*matrix)[0].size();
    if (nb.up != -1){
        MPI_Sendrecv(&((*matrix)[1][0]), rows, MPI_DOUBLE, nb.up, 0, &((*matrix)[0][0]), rows, MPI_DOUBLE, nb.up, 0, MPI_COMM_WORLD, status);
    }
    if (nb.down != -1){
        MPI_Sendrecv(&((*matrix)[columns - 2][0]), rows, MPI_DOUBLE, nb.down, 0, &((*matrix)[columns - 1][0]), rows, MPI_DOUBLE, nb.down, 0, MPI_COMM_WORLD, status);
    }
    if (nb.left != -1){
        std::vector<double> tmpSend(columns, 0);
        std::vector<double> tmpRecv(columns, 0);
        for (size_t i = 0; i < columns; i++){
            tmpSend[i] = (*matrix)[i][1];
        }
        MPI_Sendrecv(&tmpSend[0], columns, MPI_DOUBLE, nb.left, 0, &tmpRecv[0], columns, MPI_DOUBLE, nb.left, 0, MPI_COMM_WORLD, status);
        for (size_t i = 0; i < columns; i++){
            (*matrix)[i][0] = tmpRecv[i];
        }
    }
    if (nb.right != -1){
        std::vector<double> tmpSend(columns, 0);
        std::vector<double> tmpRecv(columns, 0);
        for (size_t i = 0; i < columns; i++){
            tmpSend[i] = (*matrix)[i][rows - 2];
        }
        MPI_Sendrecv(&tmpSend[0], columns, MPI_DOUBLE, nb.right, 0, &tmpRecv[0], columns, MPI_DOUBLE, nb.right, 0, MPI_COMM_WORLD, status);
        for (size_t i = 0; i < columns; i++){
            (*matrix)[i][rows - 1] = tmpRecv[i];
        }
    }
    delete status;
}

void Matrix_Out(Matrix* matrix){
    for(size_t i = 0; i < matrix->size(); i++){
        for(size_t j = 0; j < (*matrix)[i].size(); j++){
            std::cout << (*matrix)[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int isPower(int n)
{
    if (n <= 0) return(-1);
    int i = 0;
    while(n % 2 == 0) {
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

double maxNorma(Matrix *curMatrix, Matrix *prevMatrix) {
    double maxLocal = (*prevMatrix)[0][0] - (*curMatrix)[0][0];
    if (maxLocal < 0) {
        maxLocal = -maxLocal;
    }
    double tmp;
    for(int i = 1; i < curMatrix->size() - 1; i++){
        for(int j = 1; j < (*curMatrix)[0].size() - 1; j++){

        }
    }
    for (int j=0; j< curMatrix->size() - 1; j++) {
        for (int i=0; i< (*curMatrix)[0].size(); i++) {
            tmp = (*prevMatrix)[j][i] - (*curMatrix)[j][i];
            if (tmp < 0) {
                tmp = -tmp;
            }
            if (tmp > maxLocal)
                maxLocal = tmp;
        }
    }
    double maxGlobal;
    // Find max
    MPI_Allreduce(&maxLocal, &maxGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return maxGlobal;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int pid, np;
    double startTime = MPI_Wtime();
    MPI_Status *s = new MPI_Status;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int power2Np = isPower(np);
    if (power2Np < 0){
        if (pid == 0){
            std::cout << "The number of procs must be a power of 2.\n";
        }
        delete s;
        MPI_Finalize();
        return (1);
    }
    if (argc != 2){
        std::cout << "Points must be declared.\n";
        delete s;
        MPI_Finalize();
        return(2);
    }
    size_t points = atoi(argv[1]) + 1;
    if (points <= 0){
        if (pid == 0){
            std::cout << "incorrect Grid size.\n";
        }
        delete s;
        MPI_Finalize();
        return (3);
    }
    size_t processInRow = 0;
    if (power2Np % 2 == 0) {
        processInRow = pow(2, power2Np / 2);
    } else {
        processInRow = pow(2, (power2Np + 1) / 2);
    }
    size_t quotient = points / processInRow;
    size_t mod = points % processInRow;
    size_t tmpId = pid % processInRow;
    size_t processRowsQuantity = quotient + (tmpId < mod ? 1 : 0);
    size_t startRowsIndex = 0;
    if (tmpId < mod){
        startRowsIndex = (quotient + 1) * tmpId;
    }
    else{
        startRowsIndex = (quotient + 1) * mod + quotient * (tmpId - mod);
    }
    size_t processInColumns = np / processInRow;
    quotient = points / processInColumns;
    mod = points % processInColumns;
    tmpId = pid / processInRow;
    size_t processColumnsQuantity = quotient + (tmpId < mod ? 1 : 0);
    size_t startColumnsIndex = 0;
    if (tmpId < mod){
        startColumnsIndex = (quotient + 1) * tmpId;
    }
    else{
        startColumnsIndex = (quotient + 1) * mod + quotient * (tmpId - mod);
    }
    Neighbors processNeighbors = {-1, -1, -1, -1};
    if (pid / processInRow != 0){
        processNeighbors.up = pid - processInRow;
        startColumnsIndex--;
        processColumnsQuantity++;
    }
    if (pid / processInRow != processInColumns - 1){
        processNeighbors.down = pid + processInRow;
        processColumnsQuantity++;
    }
    if (pid % processInRow != 0){
        processNeighbors.left = pid - 1;
        startRowsIndex--;
        processRowsQuantity++;
    }
    if ((pid + 1) % processInRow != 0){
        processNeighbors.right = pid + 1;
        processRowsQuantity++;
    }
    std::vector<double> x(processRowsQuantity, 0);
    GridStep h;
    h.h1.resize(processRowsQuantity, 0);
    h.avg_h1.resize(processRowsQuantity, 0);
    for(size_t i = 0; i < processRowsQuantity; i++){
        x[i] = A2 * f(static_cast<double>((startRowsIndex + i)) / (points - 1), q) + A1 * (1 - f(static_cast<double>((startRowsIndex + i)) / (points - 1), q));
    }
    h.h1[0] = x[1] - x[0];
    for(size_t i = 1; i < processRowsQuantity - 1; i++){
        h.h1[i] = x[i + 1] - x[i];
        h.avg_h1[i] = 0.5 * (x[i + 1] - x[i - 1]);
    }
    std::vector<double> y(processColumnsQuantity, 0);
    h.h2.resize(processColumnsQuantity + 1, 0);
    h.avg_h2.resize(processColumnsQuantity, 0);
    for (size_t i = 0; i < processColumnsQuantity; i++){
        y[i] = B2 * f(static_cast<double>((startColumnsIndex + i)) / (points - 1), q) + B1 * (1 - f(static_cast<double>((startColumnsIndex + i)) / (points - 1), q));
    }
    h.h2[0] = y[1] - y[0];
    for(size_t i = 1; i < processColumnsQuantity - 1; i++){
        h.h2[i] = y[i + 1] - y[i];
        h.avg_h2[i] = 0.5 * (y[i + 1] - y[i - 1]);
    }
    Matrix* matrix(new Matrix());
    Matrix* rMatrix(new Matrix(processColumnsQuantity, std::vector<double>(processRowsQuantity, 0)));
    Matrix* gMatrix(new Matrix(processColumnsQuantity, std::vector<double>(processRowsQuantity, 0)));
    for(size_t i = 0; i < processColumnsQuantity; i++){
        matrix->push_back(std::vector<double>());
        for(size_t j = 0; j < processRowsQuantity; j++){
            if ((processNeighbors.up == -1) && (i == 0)){
                (*matrix)[i].push_back(phi(x[j],y[i]));
            }
            else if ((processNeighbors.down == -1) && (i == (processColumnsQuantity - 1))){
                (*matrix)[i].push_back(phi(x[j], y[i]));
            }
            else if ((processNeighbors.left == -1) && (j == 0)){
                (*matrix)[i].push_back(phi(x[j], y[i]));
            }
            else if ((processNeighbors.right == -1) && (j == (processRowsQuantity - 1))){
                (*matrix)[i].push_back(phi(x[j], y[i]));
            }
            else{
                (*matrix)[i].push_back(1); // RANDOM VALUE
            }
        }
    }
    size_t itr = -1;
    double nrm = 0;
    double MAX_NORM = 0;
    do{
        ++itr;
        double alpha = 0;
        double tau = 0;
        sendMatrixMPI(matrix, processNeighbors);
        Matrix* laplacian = Laplacian(matrix, h);
        for(size_t i = 1; i < processColumnsQuantity - 1; i++){
            for(size_t j = 1; j < processRowsQuantity - 1; j++){
                (*rMatrix)[i][j] = (*laplacian)[i][j]	- F(x[j], y[i]);
            }
        }
        delete laplacian;
        Matrix* tmpMatrix1;
        Matrix* tmpMatrix2;
        sendMatrixMPI(rMatrix, processNeighbors);
        std::vector<double> tauFraction(2, 0);
        if (itr != 0){
            laplacian = Laplacian(rMatrix, h);
            tauFraction[0] = scalarMatrixProduct(laplacian, gMatrix, h);
            delete laplacian;
            laplacian = Laplacian(gMatrix, h);
            tauFraction[1] = scalarMatrixProduct(gMatrix, laplacian, h);
            delete laplacian;
            alpha = countAlpha(pid, np, tauFraction);
        }
        tmpMatrix1 = scaleMatrix(gMatrix, -alpha);
        delete gMatrix;
        gMatrix = sumMatrix(rMatrix, tmpMatrix1);
        delete tmpMatrix1;
        sendMatrixMPI(gMatrix, processNeighbors);
        tauFraction[0] = scalarMatrixProduct(rMatrix, gMatrix, h);
        laplacian = Laplacian(gMatrix, h);
        tauFraction[1] = scalarMatrixProduct(gMatrix, laplacian, h);
        delete laplacian;
        tau = countAlpha(pid, np, tauFraction);
        tmpMatrix1 = scaleMatrix(gMatrix, -tau);
        nrm = scalarMatrixProduct(tmpMatrix1, tmpMatrix1, h);
        tmpMatrix2 = sumMatrix(matrix, tmpMatrix1);

        MAX_NORM = maxNorma(tmpMatrix2, matrix);

        delete matrix;
        matrix = tmpMatrix2;
        delete tmpMatrix1;
    }
    while(MAX_NORM > EPS);
    delete gMatrix;
    delete rMatrix;
    Matrix* errorMatrix(new Matrix(processColumnsQuantity, std::vector<double>(processRowsQuantity, 0)));
    for(size_t i = 1; i < processColumnsQuantity - 1; i++){
        for(size_t j = 1; j < processRowsQuantity - 1; j++){
            (*errorMatrix)[i][j] = (*matrix)[i][j] - phi(x[j], y[i]);
        }
    }
    double err = scalarMatrixProduct(errorMatrix, errorMatrix, h);
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
    delete errorMatrix;
    if (pid == 0){
        std::cout << "End of program";
        std::cout << "Execution Time = " << MPI_Wtime() - startTime << "\n";
        std::cout << "Iterations = " << itr << "\n";
        std::cout << "Error = " << err << "\n";
    }
    for (int i = 0; i < np; i++){
        if (pid == i){
            std::cout << pid << ":\n";
            Matrix_Out(matrix);
        }
    }
    delete matrix;
    delete s;
    MPI_Finalize();
    return 0;
}