/** 
 All of this code is written by Aman Agrawal 
 (Indian Institute of Technology, Delhi)
*/
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector> 
//#include <random>

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "time.h"

#include "genotype.h"
#include "mailman.h"
#include "arguments.h"
//#include "helper.h"
#include "storage.h"

#if SSE_SUPPORT==1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;
//Intermediate Variables
int blocksize;
int hsegsize;
double *partialsums;
double *sum_op;		
double *yint_e;
double *yint_m;
double **y_e;
double **y_m;


struct timespec t0;

clock_t total_begin = clock();
MatrixXdr pheno;
MatrixXdr mask;
MatrixXdr covariate;  
MatrixXdr Q;
MatrixXdr v1; //W^ty
MatrixXdr v2;            //QW^ty
MatrixXdr v3;    //WQW^ty
MatrixXdr new_pheno;



genotype g;
MatrixXdr geno_matrix; //(p,n)
genotype* Geno;
int k,p,n;
int k_orig;

MatrixXdr c; //(p,k)
MatrixXdr x; //(k,n)
MatrixXdr v; //(p,k)
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr sum2;
MatrixXdr sum;  
////////
//related to phenotype	
double y_sum; 
double y_mean;

options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=false;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;
bool use_cov=false; 


vector<int> len;
vector<int> Annot;
int Nbin=8;
int Nz=10;
///////

//define random vector z's
MatrixXdr  all_zb;
MatrixXdr res;
MatrixXdr XXz;
MatrixXdr Xy;
MatrixXdr yXXy;




std::istream& newline(std::istream& in)
{
    if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
        in.setstate(std::ios_base::failbit);
    }
    return in.ignore();
}


int read_cov(bool std,int Nind, std::string filename, std::string covname){
	ifstream ifs(filename.c_str(), ios::in); 
	std::string line; 
	std::istringstream in; 
	int covIndex = 0; 
	std::getline(ifs,line); 
	in.str(line); 
	string b;
	vector<vector<int> > missing; 
	int covNum=0;

	// count the number of covariates   
	while(in>>b)
	{
		if(b!="FID" && b !="IID"){
			missing.push_back(vector<int>()); //push an empty row  
			if(b==covname && covname!="")
				covIndex=covNum; 
			covNum++; 
		}
	}
	vector<double> cov_sum(covNum, 0); 
	if(covname=="")
	{
		covariate.resize(Nind, covNum); 
		cout<< "Read in "<<covNum << " Covariates.. "<<endl;
	}
	else 
	{
		covariate.resize(Nind, 1); 
		cout<< "Read in covariate "<<covname<<endl;  
	}

	
	int j=0; 
	while(std::getline(ifs, line)){ // read in a new line
		in.clear(); 
		in.str(line);
		string temp;
		in>>temp; in>>temp; //FID IID 
		for(int k=0; k<covNum; k++){// iterate through all the covariates
			
			in>>temp;
			// "NA" and number -9 are both treated as missing numbers
			if(temp=="NA")
			{
				missing[k].push_back(j);
				continue; 
			} 
			double cur = atof(temp.c_str()); 
			if(cur==-9)
			{
				missing[k].push_back(j); 
				continue; 
			}
			if(covname=="")
			{
				cov_sum[k]= cov_sum[k]+ cur; 
				covariate(j,k) = cur; 
			}
			else
				if(k==covIndex)
				{
					covariate(j, 0) = cur;
					cov_sum[k] = cov_sum[k]+cur; 
				}
		}
		j++;// increment the line
	}
	//compute cov mean and impute 
	for (int a=0; a<covNum ; a++)
	{
		int missing_num = missing[a].size();

		// calculate the average value of every covariate based on non-missing values 
		cov_sum[a] = cov_sum[a] / (Nind - missing_num);

		for(int b=0; b<missing_num; b++)
		{
			int index = missing[a][b];
			if(covname=="")
				covariate(index, a) = cov_sum[a];
			else if (a==covIndex)
				covariate(index, 0) = cov_sum[a];
		} 
	}

	if(std)// if requested to standardize the variables
	{
		MatrixXdr cov_std;
		cov_std.resize(1,covNum);  
		MatrixXdr sum = covariate.colwise().sum(); //sum of all the covariates themselves
		MatrixXdr sum2 = (covariate.cwiseProduct(covariate)).colwise().sum();// sum of squares
		MatrixXdr temp;
//		temp.resize(Nind, 1); 
//		for(int i=0; i<Nind; i++)
//			temp(i,0)=1;  
		for(int b=0; b<covNum; b++)
		{
			cov_std(0,b) = sum2(0,b) + Nind*cov_sum[b]*cov_sum[b]- 2*cov_sum[b]*sum(0,b);//EX^2-(EX)^2
			cov_std(0,b) =sqrt((Nind- 1)/cov_std(0,b)) ;
			double scalar=cov_std(0,b); 
			for(int j=0; j<Nind; j++)// standardize one by one
			{
				covariate(j,b) = covariate(j,b)-cov_sum[b];
				// divided by estimated error(notice the denominator is Nind-1)  
				covariate(j,b) =covariate(j,b)*scalar;
			} 
			//covariate.col(b) = covariate.col(b) -temp*cov_sum[b];
			
		}
	}	
	return covNum; 
}


void read_pheno2(int Nind, std::string filename){

	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	std::istringstream in;  
	int phenocount=0; 

	std::getline(ifs,line); 
	in.str(line); 
	string b; 
	while(in>>b)
	{
		if(b!="FID" && b !="IID")
			phenocount++; 
	}
	pheno.resize(Nind, phenocount);
	mask.resize(Nind, phenocount);
	int i=0;  
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		string temp;
		//fid,iid
		in>>temp; in>>temp; //skip FID and IID
		for(int j=0; j<phenocount;j++) {
			in>>temp;
			double cur = atof(temp.c_str());
			if(temp=="NA" || cur==-9){
				pheno(i,j)=0;
				mask(i,j)=0;
			}
			else{
				pheno(i,j)=atof(temp.c_str());
				mask(i,j)=1;
			}

    
		}
		i++;
	}
	//cout<<pheno; 
}


void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
	
	for(int k_iter=0;k_iter<Ncol_op;k_iter++){
		sum_op[k_iter]=op.col(k_iter).sum();		
	}

			//cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on premultiply"<<endl;
			cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
			cout << "Segment size = " << g.segment_size_hori << endl;
			cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
			cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
		}
	#endif


	//TODO: Memory Effecient SSE FastMultipy

	for(int seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply(g.segment_size_hori,g.Nindv,Ncol_op,g.p[seg_iter],op,yint_m,partialsums,y_m);
		int p_base = seg_iter*g.segment_size_hori; 
		for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++ ){
			for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
				res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
		}
	}

	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply(last_seg_size,g.Nindv,Ncol_op,g.p[g.Nsegments_hori-1],op,yint_m,partialsums,y_m);		
	int p_base = (g.Nsegments_hori-1)*g.segment_size_hori;
	for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
			res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on premultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	for(int p_iter=0;p_iter<p;p_iter++){
 		for(int k_iter=0;k_iter<Ncol_op;k_iter++){		 
			res(p_iter,k_iter) = res(p_iter,k_iter) - (g.get_col_mean(p_iter)*sum_op[k_iter]);
			if(var_normalize)
				res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter));		
 		}		
 	}	

}

void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means){

	MatrixXdr op;
	op = op_orig.transpose();

	if(var_normalize && subtract_means){
		for(int p_iter=0;p_iter<p;p_iter++){
			for(int k_iter=0;k_iter<Nrows_op;k_iter++)		
				op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter));		
		}		
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on postmultiply"<<endl;
		}
	#endif
	
	int Ncol_op = Nrows_op;

	//cout << "ncol_op = " << Ncol_op << endl;

	int seg_iter;
	for(seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply_pre(g.segment_size_hori,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);
	}
	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply_pre(last_seg_size,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);

	for(int n_iter=0; n_iter<n; n_iter++)  {
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) {
			res(k_iter,n_iter) = y_e[n_iter][k_iter];
			y_e[n_iter][k_iter] = 0;
		}
	}
	
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on postmultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	double *sums_elements = new double[Ncol_op];
 	memset (sums_elements, 0, Nrows_op * sizeof(int));

 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		double sum_to_calc=0.0;		
 		for(int p_iter=0;p_iter<p;p_iter++)		
 			sum_to_calc += g.get_col_mean(p_iter)*op(p_iter,k_iter);		
 		sums_elements[k_iter] = sum_to_calc;		
 	}		
 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		for(int n_iter=0;n_iter<n;n_iter++)		
 			res(k_iter,n_iter) = res(k_iter,n_iter) - sums_elements[k_iter];		
 	}


}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	for(int p_iter=0;p_iter<p;p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++){
			double temp=0;
			for(int n_iter=0;n_iter<n;n_iter++)
				temp+= g.get_geno(p_iter,n_iter,var_normalize)*op(n_iter,k_iter);
			res(p_iter,k_iter)=temp;
		}
	}
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	for(int n_iter=0;n_iter<n;n_iter++){
		for(int k_iter=0;k_iter<Nrows_op;k_iter++){
			double temp=0;
			for(int p_iter=0;p_iter<p;p_iter++)
				temp+= op(k_iter,p_iter)*(g.get_geno(p_iter,n_iter,var_normalize));
			res(k_iter,n_iter)=temp;
		}
	}
}

void multiply_y_pre_naive(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	res = geno_matrix * op;
}

void multiply_y_post_naive(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	res = op * geno_matrix;
}

void multiply_y_post(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_post_fast(op,Nrows_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_post_naive_mem(op,Nrows_op,res);
		else
			multiply_y_post_naive(op,Nrows_op,res);
	}
}

void multiply_y_pre(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_pre_fast(op,Ncol_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_pre_naive_mem(op,Ncol_op,res);
		else
			multiply_y_pre_naive(op,Ncol_op,res);
	}
}

void initial_var(int key)
{
    /*if(key==1)
        g=g1;
    if(key==2)
    	g=g2;*/
   // g=Geno[key];
    p = g.Nsnp;
	n = g.Nindv;


	c.resize(p,k);
	x.resize(k,n);
	v.resize(p,k);
	means.resize(p,1);
	stds.resize(p,1);
	sum2.resize(p,1); 
	sum.resize(p,1); 

	if(!fast_mode && !memory_efficient){
		geno_matrix.resize(p,n);
		g.generate_eigen_geno(geno_matrix,var_normalize);
	}

	//TODO: Initialization of c with gaussian distribution
	c = MatrixXdr::Random(p,k);


	// Initial intermediate data structures
	blocksize = k;
	hsegsize = g.segment_size_hori; 	// = log_3(n)
	int hsize = pow(3,hsegsize);		 
	int vsegsize = g.segment_size_ver; 		// = log_3(p)
	int vsize = pow(3,vsegsize);		 

	partialsums = new double [blocksize];
	sum_op = new double[blocksize];
	yint_e = new double [hsize*blocksize];
	yint_m = new double [hsize*blocksize];
	memset (yint_m, 0, hsize*blocksize * sizeof(double));
	memset (yint_e, 0, hsize*blocksize * sizeof(double));

	y_e  = new double*[g.Nindv];
	for (int i = 0 ; i < g.Nindv ; i++) {
		y_e[i] = new double[blocksize];
		memset (y_e[i], 0, blocksize * sizeof(double));
	}

	y_m = new double*[hsegsize];
	for (int i = 0 ; i < hsegsize ; i++)
		y_m[i] = new double[blocksize];
	for(int i=0;i<p;i++){
		means(i,0) = g.get_col_mean(i);
		stds(i,0) =1/g.get_col_std(i);
		//sum2(i,0) =g.get_col_sum2(i); 
		sum(i,0)= g.get_col_sum(i); 
	}

}

MatrixXdr multi_Xz (MatrixXdr zb){

	for(int j=0; j<g.Nsnp;j++)
		zb(j,0) =zb(j,0) *stds(j,0);
                                              
	MatrixXdr new_zb = zb.transpose(); 
		MatrixXdr new_res(1, g.Nindv);
	multiply_y_post_fast(new_zb, 1, new_res, false); 
	MatrixXdr new_resid(1, g.Nsnp); 
	MatrixXdr zb_scale_sum = new_zb * means;
	new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);
	new_res=new_res - new_resid;
	return new_res;

}	

double compute_yVKVy(int s){
	MatrixXdr new_pheno_sum = new_pheno.colwise().sum();
	MatrixXdr res(g.Nsnp, 1); 
	multiply_y_pre_fast(new_pheno,1,res,false); 
	res = res.cwiseProduct(stds); 
	MatrixXdr resid(g.Nsnp, 1); 
	resid = means.cwiseProduct(stds); 
	resid = resid *new_pheno_sum; 	
	MatrixXdr Xy(g.Nsnp,1); 
	Xy = res-resid; 
	double ytVKVy = (Xy.array()* Xy.array()).sum(); 
	ytVKVy = ytVKVy/g.Nsnp; 
	return ytVKVy;

}

double compute_yXXy(){

	MatrixXdr res(g.Nsnp, 1);// store the result of XT*Y
	multiply_y_pre_fast(pheno,1,res,false);
	//cout << res.transpose() << endl;
	//cout << stds.transpose() << endl;
	res = res.cwiseProduct(stds);
	MatrixXdr resid(g.Nsnp, 1);
	resid = means.cwiseProduct(stds);
	resid = resid *y_sum;
	MatrixXdr Xy(g.Nsnp,1);
	Xy = res-resid;

	double yXXy = (Xy.array()* Xy.array()).sum();
	

	return yXXy;

}





 double compute_tr_k(int s){
	  
	   /* if (s==1)
         initial_var(1);  
        if(s==2) 
         initial_var(2); 
*/
        initial_var(s);
	    double tr_k =0 ;
 		MatrixXdr temp = sum2 + g.Nindv* means.cwiseProduct(means) - 2 * means.cwiseProduct(sum);
		temp = temp.cwiseProduct(stds);
		temp = temp.cwiseProduct(stds); 
		tr_k = temp.sum() / g.Nsnp;
	   
	//    cout<<g.Nindv<<"    "<<g.Nsnp<<" s:   "<<temp.sum()<<"\n"; 
	//    cout<<tr_k<<"\n";
	   return tr_k;
	}
	
MatrixXdr  compute_XXz (){

	// cout<<mask.transpose()<<endl;
	for (int i=0;i<Nz;i++)
	    for(int j=0;j<g.Nindv;j++)
			all_zb(j,i)=all_zb(j,i)*mask(j,0);

    res.resize(g.Nsnp, Nz);
    multiply_y_pre_fast(all_zb,Nz,res, false);

    // cout<<res.transpose()<<endl;

    MatrixXdr zb_sum = all_zb.colwise().sum();
        

	for(int j=0; j<g.Nsnp; j++)
        for(int k=0; k<Nz;k++)
            res(j,k) = res(j,k)*stds(j,0);

    MatrixXdr resid(g.Nsnp, Nz);
    MatrixXdr inter = means.cwiseProduct(stds);
    resid = inter * zb_sum;
    MatrixXdr inter_zb = res - resid;
       

	for(int k=0; k<Nz; k++)
        for(int j=0; j<g.Nsnp;j++)
            inter_zb(j,k) =inter_zb(j,k) *stds(j,0);

    MatrixXdr new_zb = inter_zb.transpose();
    MatrixXdr new_res(Nz, g.Nindv);
       
    multiply_y_post_fast(new_zb, Nz, new_res, false);
       
    MatrixXdr new_resid(Nz, g.Nsnp);
    MatrixXdr zb_scale_sum = new_zb * means;
    new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);

	//return new_res;

    /// new zb 
    MatrixXdr temp=new_res - new_resid;

	for (int i=0;i<Nz;i++)
        for(int j=0;j<g.Nindv;j++)
            temp(i,j)=temp(i,j)*mask(j,0);


	return temp.transpose();
       

}


void read_annot (string filename){
    ifstream inp(filename.c_str());
    if (!inp.is_open()){
		cerr << "Error reading file "<< filename <<endl;
		exit(1);
    }
    string line;
    int j = 0 ;
    int linenum = 0 ;
    int num_parti;
    stringstream check1(line);
    string intermediate;
    vector <string> tokens;
    while(std::getline (inp, line)){
		linenum ++;
		char c = line[0];
		if (c=='#')
				continue;
		istringstream ss (line);
		if (line.empty())
				continue;
		j++;
		//cout<<line<<endl;

		stringstream check1(line);
		string intermediate;
		vector <string> tokens;
		// Tokenizing w.r.t. space ' '
		while(getline(check1, intermediate, ' '))
		{
			tokens.push_back(intermediate);
		}
		if(linenum==1){
		num_parti=tokens.size();
		if(num_parti!=Nbin)
			cout<<"number of col of annot file does not match number of bins"<<endl;
			len.resize(num_parti,0);
		}
		int index_annot=0;
		for(int i = 0; i < tokens.size(); i++){
			if (tokens[i]=="1")
				index_annot=i;
		}
		Annot.push_back(index_annot);
		len[index_annot]++;
   }

}

void count_fam(std::string filename){
	ifstream ifs(filename.c_str(), ios::in);

	std::string line;
	int i=0;
	while(std::getline(ifs, line)){
		i++;
	}
	g.Nindv=i-1;
}


int main(int argc, char const *argv[]){
  
	/*
	parse all the arguments about file directory and number of supportive vectors
	*/
	parse_args(argc,argv);
	////////////////////////////////////////////
    
	int B = command_line_opts.batchNum;
	k_orig = command_line_opts.num_of_evec ;
	debug = command_line_opts.debugmode ;
	check_accuracy = command_line_opts.getaccuracy;
	var_normalize = false;
	accelerated_em = command_line_opts.accelerated_em;
	k = k_orig + command_line_opts.l;
	k = (int)ceil(k/10.0)*10;
	command_line_opts.l = k - k_orig;
	//p = g.Nsnp;
	//n = g.Nindv;
	bool toStop=false;
	toStop=true;
	srand((unsigned int) time(0));
	//Nz=10;
	Nz=command_line_opts.num_of_evec;
	k=Nz;
	///clock_t io_end = clock();


	////
	string filename;
	
	//////////////////////////// Read multi genotypes
	string line;
	int cov_num;
	int num_files=0;
	
	string name=command_line_opts.GENOTYPE_FILE_PATH;
	ifstream f (name.c_str());
	while(getline(f,line))
		num_files++;    
   
	string file_names[num_files];

	int i=0;
	ifstream ff (name.c_str());
	while(getline(ff,line)) {
		
		file_names[i]=line;
		cout<<file_names[i]<<"\n";
		i++;
	}

	Nbin=num_files;
		
	cout<<"Number of the bins: "<<Nbin<<endl;   

	filename=command_line_opts.Annot_PATH;
	read_annot(filename);

	///reading phnotype and save the number of indvs
	
	filename=command_line_opts.PHENOTYPE_FILE_PATH;
	count_fam(filename);
	read_pheno2(g.Nindv,filename);
	cout<<"Number of Indvs :"<<g.Nindv<<endl;
	y_sum=pheno.sum();

	//read covariate
	
	std::string covfile=command_line_opts.COVARIATE_FILE_PATH;
	std::string covname="";
	if(covfile!=""){
		use_cov=true;
		cov_num=read_cov(false,g.Nindv, covfile, covname);
	}
	else if(covfile=="")
		cout<<"No Covariate File Specified"<<endl;

	/// regress out cov from phenotypes

	if(use_cov==true){
		MatrixXdr mat_mask=mask.replicate(1,cov_num);
		covariate=covariate.cwiseProduct(mat_mask);

		MatrixXdr WtW= covariate.transpose()*covariate;
		Q=WtW.inverse(); // Q=(W^tW)^-1
		MatrixXdr v1=covariate.transpose()*pheno; //W^ty
		MatrixXdr v2=Q*v1;            //QW^ty
		MatrixXdr v3=covariate*v2;    //WQW^ty
		pheno=pheno-v3;
		pheno=pheno.cwiseProduct(mask);
	}                 
	////// normalize phenotype

	//bool pheno_norm=false;
	y_sum=pheno.sum();
	y_mean = y_sum/mask.sum();

	//if(pheno_norm==true){
	for(int i=0; i<g.Nindv; i++){
		if(pheno(i,0)!=0)
			pheno(i,0) =pheno(i,0) - y_mean; //center phenotype
	}
	y_sum=pheno.sum();

	//}



	//define random vector z's
	//Nz=1;

	all_zb= MatrixXdr::Random(g.Nindv,Nz); // random uniform distribution
	all_zb = all_zb * sqrt(3); // to achieve variance of 1
	MatrixXdr output;
	
	//e
	//Njack=1;

	XXz=MatrixXdr::Zero(g.Nindv,Nbin*Nz);
	yXXy=MatrixXdr::Zero(Nbin,1);

	for(int bin_index=0; bin_index<Nbin; bin_index++){

		std::stringstream f3;
		f3 << file_names[bin_index] << ".bed";
		string name=f3.str();
		// cout<<name<<endl;
		ifstream ifs (name.c_str(), ios::in|ios::binary);
		
		g.read_header=true;
		//E
		g.Nsnp = len[bin_index];
        g.read_plink(ifs,file_names[bin_index],missing,fast_mode);
        initial_var(0);
        output=compute_XXz();

        for (int z_index=0;z_index<Nz;z_index++){
            XXz.col((bin_index*Nz)+z_index)=output.col(z_index);
            XXz.col((bin_index*Nz)+z_index)+=output.col(z_index);   /// save whole sample
        }

        yXXy(bin_index,0)= compute_yXXy();

        ///////end computation
        /////////////////////////////////destruct class g
        delete[] sum_op;
        delete[] partialsums;
        delete[] yint_e;
        delete[] yint_m;
        for (int i  = 0 ; i < hsegsize; i++)
            delete[] y_m [i];
        delete[] y_m;

        for (int i  = 0 ; i < g.Nindv; i++)
            delete[] y_e[i];
        delete[] y_e;

        std::vector< std::vector<int> >().swap(g.p);
        std::vector< std::vector<int> >().swap(g.not_O_j);
        std::vector< std::vector<int> >().swap(g.not_O_i);

        //g.p.clear();
        //g.not_O_j.clear();
        //g.not_O_i.clear();
        g.columnsum.clear();
        g.columnsum2.clear();
        g.columnmeans.clear();
        g.columnmeans2.clear();
        //std::vector< std::vector<int> >().swap(g.columnsum);
        //std::vector< std::vector<int> >().swap(g.columnsum2);
        //std::vector< std::vector<double> >().swap(g.columnmeans);
        //std::vector< std::vector<double> >().swap(g.columnmeans2);
        g.read_header=false;
        /////////////////////////////////////////////


			
		if(bin_index==0){
			// exception handling when bin is zero
			// yet to finish
		}
	
	} //end of loop over bins


	/// normal equations LHS
	MatrixXdr  A_trs(Nbin,Nbin);
	MatrixXdr b_trk(Nbin,1);
	MatrixXdr c_yky(Nbin,1);

	MatrixXdr X_l(Nbin+1,Nbin+1);
	MatrixXdr Y_r(Nbin+1,1);
	//int bin_index=0;
	MatrixXdr B1;
	MatrixXdr B2;
	MatrixXdr C1;
	MatrixXdr C2;
	double trkij;
	double yy=(pheno.array() * pheno.array()).sum();
	int Nindv_mask=mask.sum();
	MatrixXdr jack;
	MatrixXdr point_est;
	MatrixXdr enrich_jack;
	MatrixXdr enrich_point_est;

	point_est.resize(Nbin+1,1);

	enrich_point_est.resize(Nbin,1);




    for (int i=0;i<Nbin;i++){

        b_trk(i,0)=Nindv_mask;
        c_yky(i,0)=yXXy(i,0)/len[i];

        for (int j=i;j<Nbin;j++){
            //cout<<Njack<<endl;
            B1=XXz.block(0,(i*Nz),g.Nindv,Nz);
            B2=XXz.block(0,(j*Nz),g.Nindv,Nz);
            C1=B1.array()*B2.array();
            C2=C1.colwise().sum();
            trkij=C2.sum();
            trkij=trkij/len[i]/len[j]/Nz;
            A_trs(i,j)=trkij;
            A_trs(j,i)=trkij;

        }
    }

    X_l<<A_trs,b_trk,b_trk.transpose(),Nindv_mask;
    Y_r<<c_yky,yy;

    MatrixXdr herit=X_l.colPivHouseholderQr().solve(Y_r);

    for(int i=0;i<(Nbin+1);i++)
        point_est(i,0)=herit(i,0);

    double total_val=0;
    for(int i=0; i<Nbin;i++)
        total_val+=herit(i,0);

	double temp_sig=0;
	double temp_sum=0;

	temp_sig=0;
	temp_sum=point_est.sum();
	for (int j=0;j<Nbin;j++){
		point_est(j,0)=point_est(j,0)/temp_sum;
		temp_sig+=point_est(j,0);
	}
	point_est(Nbin,0)=temp_sig;


	///compute enrichment

	double per_her;
	double per_size;
	int total_size=0;

	for (int i=0;i<Nbin;i++)
    	total_size+=len[i];

	for (int j=0;j<Nbin;j++){
        per_her=point_est(j,0)/point_est(Nbin,0);
        per_size=(double)len[j]/total_size;
        enrich_point_est(j,0)=per_her/per_size;
	}

	//for (int i=0;i<Njack;i++)
	//  cout<<jack.col(i).transpose()<<endl;
	cout<<"OUTPUT: "<<endl;
	for (int j=0;j<Nbin;j++)
		cout<<"h^2 of bin "<<j<<" : "<<point_est(j,0)<<endl;
	cout<<"Total h^2 : "<<point_est(Nbin,0)<<endl;
	for (int j=0;j<Nbin;j++)
		cout<<"Enrichment of bin "<<j<<" :"<<enrich_point_est(j,0)<<endl;

	std::ofstream outfile;
	string add_output=command_line_opts.OUTPUT_FILE_PATH;
	outfile.open(add_output.c_str(), std::ios_base::app);

	outfile<<"Point estimates :"<<endl;
	outfile<<point_est.transpose()<<endl;
//	outfile<<"SEs     :"<<endl;
//	outfile<<SEjack.transpose()<<endl;
		
	outfile<<"Enrichment :"<<endl;
	outfile<<enrich_point_est.transpose()<<endl;
//	outfile<<"SEs     :"<<endl;
//	outfile<<enrich_SEjack.transpose()<<endl;

	return 0;
}
