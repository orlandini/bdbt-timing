/**
 * @file
 * @brief Contains implementations of the TPZMatMassMatrix methods.
 */

#include "TPZMatMassMatrix.h"
#include "Material/pzbndcond.h"

TPZMatMassMatrix::TPZMatMassMatrix(int nummat, int dim) : TPZRegisterClassId(&TPZMatMassMatrix::ClassId),
TPZMaterial(nummat), fXf(0.), fDim(dim), fK(1){
#ifdef PZDEBUG
    if(dim < 1 || dim > 3)
    {
        DebugStop();
    }
#endif
}

TPZMatMassMatrix::TPZMatMassMatrix():TPZRegisterClassId(&TPZMatMassMatrix::ClassId),
TPZMaterial(), fXf(0.), fDim(1), fK(1){
}

TPZMatMassMatrix::TPZMatMassMatrix(const TPZMatMassMatrix &copy):TPZRegisterClassId(&TPZMatMassMatrix::ClassId),
TPZMaterial(copy){
	this->operator =(copy);
}

TPZMatMassMatrix & TPZMatMassMatrix::operator=(const TPZMatMassMatrix &copy){
	TPZMaterial::operator = (copy);
	fDim = copy.fDim;
	fXf  = copy.fXf;
	fK   = copy.fK;
	return *this;
}

void TPZMatMassMatrix::SetDiffusiveParameter(STATE diff) {
	fK = diff;
}

void TPZMatMassMatrix::GetDiffusiveParameter(STATE &diff) {
	diff = fK;
}

TPZMatMassMatrix::~TPZMatMassMatrix() {
}

void TPZMatMassMatrix::Print(std::ostream &out) {
	out << "name of material : " << Name() << "\n";
	out << "Laplace operator multiplier fK "<< fK << std::endl;
	out << "Forcing vector fXf " << fXf << std::endl;
	out << "Base Class properties :";
	TPZMaterial::Print(out);
	out << "\n";
}

void TPZMatMassMatrix::Contribute(TPZMaterialData &data,REAL weight,TPZFMatrix<STATE> &ek,TPZFMatrix<STATE> &ef) {
    const TPZFMatrix<REAL> & phi = data.phi;
    const auto phr = phi.Rows();
    //Poisson eq
    for(auto in = 0; in < phr; in++ ) {
        for(auto jn = 0; jn < phr; jn++ ) {
            ek(in,jn) += (STATE)weight * fK * (STATE)( phi.GetVal(in,0) * phi.GetVal(jn,0) ) ;
        }
    }
//    const TPZFMatrix<REAL> & phi = data.phi;
//    const TPZFMatrix<REAL> & dphi = data.dphix;
//
//    TPZVec<REAL>  &x = data.x;
//    const auto phr = dphi.Cols();
//
//    const STATE fXfLoc = [this,x](){
//	    if(fForcingFunction) {            // phi(in, 0) = phi_in
//	        TPZManVector<STATE,1> res(1);
//	        TPZFMatrix<STATE> dres(Dimension(),1);
//	        fForcingFunction->Execute(x,res,dres);       // dphi(i,j) = dphi_j/dxi
//	        return res[0];
//	    }
//	    else{
//	    	return fXf;
//	    }
//    }();
//
//    //Poisson eq
//    for(auto in = 0; in < phr; in++ ) {
//        ef(in, 0) += (STATE)weight * fXfLoc * phi.GetVal(in,0);
//        for(auto jn = 0; jn < phr; jn++ ) {
//            for(auto kd=0; kd<fDim; kd++) {
//                ek(in,jn) += (STATE)weight * fK * (STATE)( dphi.GetVal(kd,in) * dphi.GetVal(kd,jn) ) ;
//            }
//        }
//    }
}

void TPZMatMassMatrix::ContributeBC(TPZMaterialData &data,REAL weight,
								   TPZFMatrix<STATE> &ek,TPZFMatrix<STATE> &ef,TPZBndCond &bc) {
	
	
	const TPZFMatrix<REAL> &phi = data.phi;
	const auto phr = phi.Rows();
	const STATE v2 = [&](){
		if(bc.HasForcingFunction()) {
			TPZManVector<STATE,1> res(1);
			bc.ForcingFunction()->Execute(data.x,res);       // dphi(i,j) = dphi_j/dxi
			return res[0];
		}else{
			return bc.Val2()(0,0);
		}
	}();
	

	switch (bc.Type()) {
		case 0 :			// Dirichlet condition
			for(auto in = 0 ; in < phr; in++) {
				ef(in,0) += (STATE)(gBigNumber* phi.GetVal(in,0) * weight) * v2;
				for (auto jn = 0 ; jn < phr; jn++) {
					ek(in,jn) += gBigNumber * phi.GetVal(in,0) * phi.GetVal(jn,0) * weight;
				}
			}
			break;
		case 1 :			// Neumann condition
			for(auto in = 0 ; in < phr; in++) {
				ef(in,0) += v2 * (STATE)(phi.GetVal(in,0) * weight);
			}
			break;
		case 2 :		// mixed condition
			for(auto in = 0 ; in < phr; in++) {
				ef(in, 0) += v2 * (STATE)(phi.GetVal(in, 0) * weight);
				for (auto jn = 0 ; jn < phr; jn++) {
					ek(in,jn) += bc.Val1()(0,0) * (STATE)(phi.GetVal(in,0) * phi.GetVal(jn,0) * weight);     // peso de contorno => integral de contorno
				}
			}
			break;
			default:
		DebugStop();
	}
}

/** Returns the variable index associated with the name */
int TPZMatMassMatrix::VariableIndex(const std::string &name){
	if(!strcmp("Solution",name.c_str()))        return  1;
	if(!strcmp("Derivative",name.c_str()))      return  2;
	if(!strcmp("KDuDx",name.c_str()))           return  3;
	if(!strcmp("KDuDy",name.c_str()))           return  4;
	if(!strcmp("KDuDz",name.c_str()))           return  5;
	if(!strcmp("NormKDu",name.c_str()))         return  6;
	if(!strcmp("MinusKGradU",name.c_str()))     return  7;
	if(!strcmp("POrder",name.c_str()))          return  8;
	if(!strcmp("Laplac",name.c_str()))          return  9;
	if(!strcmp("Stress",name.c_str()))          return  10;    
	if(!strcmp("Flux",name.c_str()))            return  10;
	if(!strcmp("Pressure",name.c_str()))        return  11;
	
	if (!strcmp("ExactPressure", name.c_str()))   return  12;
	if(!strcmp("ExactSolution",name.c_str()))   return  12;
	if(!strcmp("ExactFlux",name.c_str()))       return  13;
	if(!strcmp("Divergence",name.c_str()))      return  14;
	if(!strcmp("ExactDiv",name.c_str()))        return  15;
	
	if(!strcmp("PressureOmega1",name.c_str()))  return  16;
	if(!strcmp("PressureOmega2",name.c_str()))  return  17;
	if(!strcmp("FluxOmega1",name.c_str()))      return  18;
    
    if(!strcmp("GradFluxX",name.c_str()))       return  19;
    if(!strcmp("GradFluxY",name.c_str()))       return  20;
	return TPZMaterial::VariableIndex(name);
}

int TPZMatMassMatrix::NSolutionVariables(int var){
	if(var == 1) return 1;
	if(var == 2) return fDim;//arrumar o fluxo de hdiv para ser fdim tbem enquanto isso faco isso
	if ((var == 3) || (var == 4) || (var == 5) || (var == 6)) return 1;
	if (var == 7) return fDim;
	if (var == 8) return 1;
	if (var == 9) return 1;
	if (var==10) return fDim;
	if (var==11) return 1;
	
	if (var==12) return 1;
	if (var==13) return fDim;
	if (var==14) return 1;
	if (var==15) return 1;
	//teste de acoplamento
	if (var==16) return 1;
	if (var==17) return 1;
	if (var==18) return 3;
    if (var==19) return 3;
    if (var==20) return 3;
    if (var==21) return fDim;
	
	
	return TPZMaterial::NSolutionVariables(var);
}

void TPZMatMassMatrix::Solution(TPZMaterialData &data, int var, TPZVec<STATE> &Solout){
	
	TPZVec<STATE> pressure(1);
	TPZVec<REAL> pt(3);
	TPZFMatrix<STATE> flux(3,1);
	
    int numbersol = data.sol.size();
    if (numbersol != 1) {
        DebugStop();
    }
	
   // Solout.Resize(this->NSolutionVariables(var));
    
#ifndef STATE_COMPLEX
	
	switch (var) {
		case 8:
			Solout[0] = data.p;
			break;
		case 10:
			if (data.numberdualfunctions) {
				
				Solout[0]=data.sol[0][0];
				Solout[1]=data.sol[0][1];
                Solout[2]=data.sol[0][2];
				
			}
			else {
				this->Solution(data.sol[0], data.dsol[0], data.axes, 2, Solout);
			}
			
			break;
            
        case 21:
            for(int k=0;k<fDim;k++){
                Solout[k]=data.sol[0][k];
            }
			break;
            
		case 11:
			if (data.numberdualfunctions) {
				Solout[0]=data.sol[0][2];
			}
			else{
				Solout[0]=data.sol[0][0];
			}
			break;
			
		case 12:
				fForcingFunctionExact->Execute(data.x,pressure,flux);
				
				Solout[0]=pressure[0];
			break;
		case 13:
				fForcingFunctionExact->Execute(data.x,pressure,flux);
				
				Solout[0]=flux(0,0);
				Solout[1]=flux(1,0);
            break;
            
        case 14:
        {
			if (data.numberdualfunctions){
				Solout[0]=data.sol[0][data.sol[0].NElements()-1];
			}else{
                //Solout[0]=data.dsol[0](0,0)+data.dsol[0](1,1)+data.dsol[0](2,2);
                STATE val = 0.;
                for(int i=0; i<fDim; i++){
                    val += data.dsol[0](i,i);
                }
                Solout[0] = val;
            }
        }
            break;
          
        case 15:
        {
            fForcingFunctionExact->Execute(data.x,pressure,flux);
            Solout[0]=flux(fDim,0);
        }
            break;

            
        case 16:
            if (data.numberdualfunctions) {
					Solout[0]=data.sol[0][2];
            }
            else {
                std::cout<<"Pressao somente em Omega1"<<std::endl;
                Solout[0]=0;//NULL;
            }
				
            break;
        
        case 17:
            if (!data.numberdualfunctions) {
                Solout[0]=data.sol[0][0];
            }
            else {
                std::cout<<"Pressao somente em omega2"<<std::endl;
                Solout[0]=0;//NULL;
            }
				
            break;
        case 18:
            if( data.numberdualfunctions){
                Solout[0]=data.sol[0][0];//fluxo de omega1
                Solout[1]=data.sol[0][1];
                //	Solout[2]=data.sol[2];
                return;
            }
        
        case 19:
            if(data.numberdualfunctions){
                Solout[0]=data.dsol[0](0,0);//fluxo de omega1
                Solout[1]=data.dsol[0](1,0);
                Solout[2]=data.dsol[0](2,0);
                return;
            }
        case 20:
            if( data.numberdualfunctions){
                Solout[0]=data.dsol[0](0,1);//fluxo de omega1
                Solout[1]=data.dsol[0](1,1);
                Solout[2]=data.dsol[0](2,1);
                return;
            }
            else {
                std::cout<<"Pressao somente em omega2"<<std::endl;
                Solout[0]=0;//NULL;
            }
            break;
        default:
           
            if (data.sol[0].size() == 4) {
                
                data.sol[0][0] = data.sol[0][2];
            }

            this->Solution(data.sol[0], data.dsol[0], data.axes, var, Solout);
            break;
    }
#endif
}

#include "pzaxestools.h"
void TPZMatMassMatrix::Solution(TPZVec<STATE> &Sol,TPZFMatrix<STATE> &DSol,TPZFMatrix<REAL> &axes,int var,TPZVec<STATE> &Solout){
	
#ifndef STATE_COMPLEX
	Solout.Resize( this->NSolutionVariables( var ) );
	
	if(var == 1){
		Solout[0] = Sol[0];//function
		return;
	}
	if(var == 2) {
		int id;
		for(id=0 ; id<fDim; id++) {
			TPZFNMatrix<9,STATE> dsoldx;
			TPZAxesTools<STATE>::Axes2XYZ(DSol, dsoldx, axes);
			Solout[id] = dsoldx(id,0);//derivate
		}
		return;
	}//var == 2
	if (var == 3){ //KDuDx
		TPZFNMatrix<9,STATE> dsoldx;
		TPZAxesTools<STATE>::Axes2XYZ(DSol, dsoldx, axes);
		Solout[0] = dsoldx(0,0) * this->fK;
		return;
	}//var ==3
	if (var == 4){ //KDuDy
		TPZFNMatrix<9,STATE> dsoldx;
		TPZAxesTools<STATE>::Axes2XYZ(DSol, dsoldx, axes);
		Solout[0] = dsoldx(1,0) * this->fK;
		return;
	}//var == 4 
	if (var == 5){ //KDuDz
		TPZFNMatrix<9,STATE> dsoldx;
		TPZAxesTools<STATE>::Axes2XYZ(DSol, dsoldx, axes);
		Solout[0] = dsoldx(2,0) * this->fK;
		return;
	}//var == 5
	if (var == 6){ //NormKDu
		int id;
		REAL val = 0.;
		for(id=0 ; id<fDim; id++){
			val += (DSol(id,0) * this->fK) * (DSol(id,0) * this->fK);
		}
		Solout[0] = sqrt(val);
		return;
	}//var == 6
	if (var == 7){ //MinusKGradU
		int id;
		//REAL val = 0.;
		TPZFNMatrix<9,STATE> dsoldx;
		TPZAxesTools<STATE>::Axes2XYZ(DSol, dsoldx, axes);
		for(id=0 ; id<fDim; id++) {
			Solout[id] = -1. * this->fK * dsoldx(id,0);
		}
		return;
	}//var == 7  
	if(var == 9){//Laplac
		Solout.Resize(1);
		Solout[0] = DSol(2,0);
		return;
	}//Laplac
	
#endif
	TPZMaterial::Solution(Sol, DSol, axes, var, Solout);
	
}//method

void TPZMatMassMatrix::Errors(TPZVec<REAL> &x,TPZVec<STATE> &u,
							 TPZFMatrix<STATE> &dudx, TPZFMatrix<REAL> &axes, TPZVec<STATE> &/*flux*/,
							 TPZVec<STATE> &u_exact,TPZFMatrix<STATE> &du_exact,TPZVec<REAL> &values) {
	
	values.Resize(NEvalErrors());
    values.Fill(0.0);
	TPZManVector<STATE> dudxEF(1,0.), dudyEF(1,0.),dudzEF(1,0.);
	this->Solution(u,dudx,axes,VariableIndex("KDuDx"), dudxEF);
    STATE fraq = dudxEF[0]/fK;
    fraq = fraq - du_exact(0,0);
    REAL diff = fabs(fraq);
	values[3] = diff*diff;
	if(fDim > 1) {
		this->Solution(u,dudx,axes, this->VariableIndex("KDuDy"), dudyEF);
        fraq = dudyEF[0]/fK;
        fraq = fraq - du_exact(1,0);
		diff = fabs(fraq);
		values[4] = diff*diff;
		if(fDim > 2) {
			this->Solution(u,dudx,axes, this->VariableIndex("KDuDz"), dudzEF);
			fraq = dudzEF[0]/fK;
            fraq = fraq - du_exact(2,0);
            diff = fabs(fraq);
			values[5] = diff*diff;
		}
	}
	
	TPZManVector<STATE,3> sol(1),dsol(3,0.);
	Solution(u,dudx,axes,1,sol);
	Solution(u,dudx,axes,2,dsol);
	int id;
	//values[1] : eror em norma L2
    diff = fabs(sol[0] - u_exact[0]);
	values[1]  = diff*diff;
	//values[2] : erro em semi norma H1
	values[2] = 0.;
	for(id=0; id<fDim; id++) {
        diff = fabs(dsol[id] - du_exact(id,0));
		values[2]  += abs(fK)*diff*diff;
	}
	//values[0] : erro em norma H1 <=> norma Energia
	values[0]  = values[1]+values[2];
}

void TPZMatMassMatrix::Write(TPZStream &buf, int withclassid) const{
	TPZMaterial::Write(buf, withclassid);
	buf.Write(&fXf, 1);
	buf.Write(&fDim, 1);
	buf.Write(&fK, 1);
}

void TPZMatMassMatrix::Read(TPZStream &buf, void *context){
	TPZMaterial::Read(buf, context);
	buf.Read(&fXf, 1);
	buf.Read(&fDim, 1);
	buf.Read(&fK, 1);
}

int TPZMatMassMatrix::ClassId() const{
    return Hash("TPZMatMassMatrix") ^ TPZMaterial::ClassId() << 1;
}
template class TPZRestoreClass<TPZMatMassMatrix>;
