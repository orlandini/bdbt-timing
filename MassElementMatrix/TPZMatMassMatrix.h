/**
 * @file
 * @brief Contains the TPZMatMassMatrix class.
  */

#ifndef MATMASSMATRIX
#define MATMASSMATRIX

#include "Material/TPZMaterial.h"

/**
 * @ingroup material
 * @brief DESCRIBE PLEASE
 */
/**
 * \f$ -div(K(grad(u)) = - fXf  \f$
 */
class TPZMatMassMatrix : public TPZMaterial {
	
protected :
	
	/** @brief Forcing function value */
	STATE fXf;
	
	/** @brief Problem dimension */
	int fDim;
	
	/** @brief Coeficient which multiplies the Laplacian operator. */
	STATE fK;
	
public:
	
	TPZMatMassMatrix();

	TPZMatMassMatrix(int matid) : TPZRegisterClassId(&TPZMatMassMatrix::ClassId),
    TPZMaterial(matid), fXf(0.), fDim(-1), fK(0.){
    
    }
    
	TPZMatMassMatrix(int matid, int dim);
	
	
	TPZMatMassMatrix(const TPZMatMassMatrix &copy);
	
	virtual ~TPZMatMassMatrix();
	
	TPZMatMassMatrix &operator=(const TPZMatMassMatrix &copy);
	
	virtual TPZMaterial * NewMaterial() override 
	{
		return new TPZMatMassMatrix(*this);
	}
    
    /** 
	 * @brief Fill material data parameter with necessary requirements for the
	 * @since April 10, 2007
	 */
	/** 
	 * Contribute method. Here, in base class, all requirements are considered as necessary. 
	 * Each derived class may optimize performance by selecting only the necessary data.
     */
    virtual void FillDataRequirements(TPZMaterialData &data) override
    {
        data.SetAllRequirements(false);
    }
	    
    /** @brief This method defines which parameters need to be initialized in order to compute the contribution of the boundary condition */
    virtual void FillBoundaryConditionDataRequirement(int type,TPZMaterialData &data) override
    {
        data.SetAllRequirements(false);
    }
    
	
	int Dimension() const override
	{ 
		return fDim;
	}
	
    /** @brief Returns the number of state variables associated with the material */
	virtual int NStateVariables() const override
	{
        return 1;
    }
	
	virtual void SetDiffusiveParameter(STATE diff);
	
	void GetDiffusiveParameter(STATE &diff);
    
    void SetDimension(int dim)
    {
#ifdef PZDEBUG
        if(dim<0 || dim >3){
            DebugStop();
        }
#endif
        fDim = dim;
    }
	
	void SetInternalFlux(STATE flux)
	{
		fXf = flux;
	}
	
	
	virtual void Print(std::ostream & out) override;
	
	virtual std::string Name() override
	{
		return "TPZMatMassMatrix";
	}

	/**
	 * @name Contribute methods (weak formulation)
	 * @{
	 */
	virtual void Contribute(TPZMaterialData &data,REAL weight,TPZFMatrix<STATE> &ek,TPZFMatrix<STATE> &ef) override;

	virtual void ContributeBC(TPZMaterialData &data,REAL weight,
							  TPZFMatrix<STATE> &ek,TPZFMatrix<STATE> &ef,TPZBndCond &bc) override;
	   
	virtual int VariableIndex(const std::string &name) override;
	
	virtual int NSolutionVariables(int var) override;
	
	virtual int NFluxes() override { return 3;}

	virtual int ClassId() const override;
	
	virtual void Write(TPZStream &buf, int withclassid) const override;
	
	virtual void Read(TPZStream &buf, void *context) override;
	
	virtual void Solution(TPZMaterialData &data, int var, TPZVec<STATE> &Solout) override;
	
	virtual void Errors(TPZVec<REAL> &x,TPZVec<STATE> &u,
				TPZFMatrix<STATE> &dudx, TPZFMatrix<REAL> &axes, TPZVec<STATE> &flux,
				TPZVec<STATE> &u_exact,TPZFMatrix<STATE> &du_exact,TPZVec<REAL> &values) override;
	
	virtual int NEvalErrors()  override {return 3;}

protected:    
	virtual void Solution(TPZVec<STATE> &Sol,TPZFMatrix<STATE> &DSol,TPZFMatrix<REAL> &axes,int var,TPZVec<STATE> &Solout) override;	
};

#endif
