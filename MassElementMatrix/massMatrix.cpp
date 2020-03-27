#include "Analysis/pzanalysis.h"
#include "Material/pzbndcond.h"
#include "Mesh/pzgmesh.h"
#include "Mesh/TPZGeoMeshTools.h"
#include "Mesh/pzintel.h"
#include "Mesh/TPZCompMeshTools.h"
#include "Pre/TPZGmshReader.h"
#include "Post/TPZVTKGeoMesh.h"
#include "Shape/pzshapelinear.h"
#include "StrMatrix/TPZSSpStructMatrix.h"
#include "StrMatrix/TPZSpStructMatrix.h"

#include "TPZMatMassMatrix.h"

#include <chrono>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <limits>
#include <string>

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/HybridMatrix.h>
#include <blaze/math/DiagonalMatrix.h>
using blaze::rowMajor;
using blaze::columnMajor;

enum EOrthogonalFuncs{
    EChebyshev = 0,EExpo = 1,ELegendre = 2 ,EJacobi = 3,EHermite = 4
};

/**
* Generates a computational mesh that implements the problem to be solved
*/
static TPZCompMesh *CreateCompMesh(TPZGeoMesh *gmesh, const TPZVec<int> &matIds, const int initialPOrder, EOrthogonalFuncs familyType);

/**
 * This method is responsible to removing the equation corresponding to dirichlet boundary conditions. Used
 * if one desires to analyse the condition number of the resultant matrix
 */
static void FilterBoundaryEquations(TPZCompMesh *cmesh, TPZVec<int64_t> &activeEquations, int64_t &neq,
                                    int64_t &neqOriginal);


static void ForcingFunction2D(const TPZVec<REAL>& pt, TPZVec<STATE> &result){
    REAL x = pt[0];
    REAL y = pt[1];
    result[0] = 1;
}
static constexpr int pOrderForcingFunction{0};

int main(int argc, char **argv)
{
    static const std::chrono::time_point<std::chrono::system_clock> wall_time_start = std::chrono::system_clock::now();
    //physical dimension of the problem
    constexpr int dim{3};
    //number of divisions of each direction (x, y or x,y,z) of the domain
    constexpr int nDiv{4};
    //initial polynomial order
    constexpr int initialPOrder{4};
    //number of h refinements
    constexpr int nHRefinements{2};
    //whether to remove the dirichlet boundary conditions from the matrix
    constexpr bool filterBoundaryEqs{true};
    //whether to export .vtk files
    constexpr bool postProcess{false};
    //which family of polynomials to use
    EOrthogonalFuncs orthogonalPolyFamily = EChebyshev;//EChebyshev = 0,EExpo = 1,ELegendre = 2 ,EJacobi = 3,EHermite = 4

    constexpr MMeshType meshType{MMeshType::ETetrahedral};

    constexpr bool useGmshMesh{true}; 
    TPZVec<int> matIdVec(dim*2+1,2);
    matIdVec[0] = 1;//volumetric elements have a different identifier
    TPZGeoMesh *gMesh = [&]() -> TPZGeoMesh *{
        if(useGmshMesh){
            TPZGmshReader gmshReader;
            gmshReader.SetFormatVersion("3.0");
            return gmshReader.GeometricGmshMesh("unitcube.msh",nullptr);

        }else{
            TPZManVector<REAL,3> minX(3,0);
            TPZManVector<REAL,3> maxX(3,1);
            TPZVec<int> nDivs(dim,nDiv);
            return TPZGeoMeshTools::CreateGeoMeshOnGrid(dim,minX,maxX,matIdVec,nDivs,meshType,true);
        }
    }();

    for(auto ih = 0; ih < nHRefinements; ih++){
        std::cout<<"Refining mesh..."<<std::endl;
        const int nel = gMesh->NElements();
        TPZManVector<TPZGeoEl *, 6> sons;
        for(auto iel = 0; iel < nel; iel++){
            TPZGeoEl *gel = gMesh->ElementVec()[iel];
            if(gel->HasSubElement()) continue;
            gel->Divide(sons);
        }
        gMesh->BuildConnectivity();
        std::cout<<"Mesh refined!"<<std::endl;
    }

    TPZCompMesh *cMesh = CreateCompMesh(gMesh,matIdVec,initialPOrder,orthogonalPolyFamily);
    //aux funcs
    auto WallTime = [] () noexcept
    {
        std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = now-wall_time_start;
        return elapsed_seconds.count();
    };

    double seconds_per_tick = [WallTime] () noexcept
    {
        auto tick_start = __rdtsc();
        double tstart = WallTime();
        double tend = WallTime()+0.001;

        // wait for 1ms and compare wall time with time counter
        while(WallTime()<tend);

        auto tick_end = __rdtsc();
        tend = WallTime();

        return (tend-tstart)/static_cast<double>(tick_end-tick_start);
    }();
    std::cout<<std::endl<<std::endl<<std::endl;
    std::cout<<"Computing Element Matrices using traditional CalcStiff"<<std::endl;
    {
        STATE res{0};
        const auto assemble_tick_b = __rdtsc();
        for(const auto& cel : cMesh->ElementVec()){
            if(cel->Dimension() != dim) continue;
            TPZElementMatrix ek, ef;
            cel->CalcStiff(ek,ef);
            res+=ek.fMat(0,0);
        }
        const auto assemble_tick_e = __rdtsc();
        std::cout<<std::setw(8)<<"\tsum ek(0,0) = "<<res<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
        std::cout <<"\tTotal time: "<<((double)(assemble_tick_e-assemble_tick_b))*seconds_per_tick<<std::endl;
        std::cout <<"\tTotal MClock Cycles: "<<(assemble_tick_e-assemble_tick_b)/1e6<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
    }
    std::cout<<"Computing Element Matrices using BDBt (naive matrix mult)"<<std::endl;
    {
        STATE res{0};
        const auto assemble_tick_b = __rdtsc();
        for(const auto& iCel : cMesh->ElementVec()){
            const auto cel = dynamic_cast<TPZInterpolationSpace *>(iCel);
            if(iCel->Dimension() != dim || cel == nullptr) continue;
            TPZElementMatrix ek, ef;
            TPZFNMatrix<60,STATE>phi;
            TPZFNMatrix<180,STATE>dphi;
            TPZFNMatrix<10000, STATE> bMat;
            TPZFNMatrix<25000, STATE> dMat;
            auto material = dynamic_cast<TPZMatMassMatrix *>(cel->Material());
            cel->InitializeElementMatrix(ek,ef);
            if (cel->NConnects() == 0) break;

            TPZAutoPointer<TPZIntPoints> intrule = cel->GetIntegrationRule().Clone();
            const auto n_int_points = intrule->NPoints();
            const auto dim1 = cel->NShapeF();
            const auto dim2 = n_int_points;
            bMat.Redim(dim1,dim2);
            dMat.Resize(dim2,dim2);

            phi.Resize(dim1,1);
            dphi.Resize(3,dim1);
            const int celDim = cel->Dimension();
            TPZManVector<REAL,3> intpoint(celDim,0.);
            REAL weight = 0.;
            STATE constitutiveParam = 0;
            material->GetDiffusiveParameter(constitutiveParam);
            REAL detjac{0};
            TPZFMatrix<REAL> jac(dim,dim),axes(dim,dim),jacinv(dim,dim);
            for(int ipt = 0; ipt < dim2; ++ipt){
                intrule->Point(ipt,intpoint,weight);
                cel->Shape(intpoint,phi,dphi);
                cel->Reference()->Jacobian(intpoint,jac,axes,detjac,jacinv);
                dMat(ipt,ipt) = weight * fabs(detjac) * constitutiveParam;
                memcpy(&bMat.Adress()[ipt*dim1],phi.Adress(),dim1*sizeof(STATE));
            }
            auto &mat = ek.fMat;
            for(auto it_j = 0; it_j < dim1; it_j ++){
                for(auto it_i = 0; it_i < dim1; it_i ++){
                    STATE val{0};
                    for(auto it_k = 0; it_k < dim2; it_k ++){
                        val += bMat.GetVal(it_i,it_k) * dMat.GetVal(it_k,it_k)* bMat.GetVal(it_j,it_k);
                    }
                    mat.PutVal(it_i,it_j,val);
                }
            }
            res+=ek.fMat(0,0);
        }
        const auto assemble_tick_e = __rdtsc();
        std::cout<<std::setw(8)<<"\tsum ek(0,0) = "<<res<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
        std::cout <<"\tTotal time: "<<((double)(assemble_tick_e-assemble_tick_b))*seconds_per_tick<<std::endl;
        std::cout <<"\tTotal MClock Cycles: "<<(assemble_tick_e-assemble_tick_b)/1e6<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
    }

    std::cout<<"Computing Element Matrices using BDBt (blaze::DynamicMatrix)"<<std::endl;
    {
        STATE res{0};
        const auto assemble_tick_b = __rdtsc();
        for(const auto& iCel : cMesh->ElementVec()){
            const auto cel = dynamic_cast<TPZInterpolationSpace *>(iCel);
            if(iCel->Dimension() != dim || cel == nullptr) continue;
            TPZFNMatrix<60,STATE>phi;
            TPZFNMatrix<180,STATE>dphi;
            auto material = dynamic_cast<TPZMatMassMatrix *>(cel->Material());
            if (cel->NConnects() == 0) break;

            TPZAutoPointer<TPZIntPoints> intrule = cel->GetIntegrationRule().Clone();
            const auto n_int_points = intrule->NPoints();
            const auto dim1 = cel->NShapeF();
            const auto dim2 = n_int_points;

            blaze::DynamicMatrix<STATE,blaze::columnMajor> bMat(dim1,dim2);
            blaze::DiagonalMatrix< blaze::DynamicMatrix<STATE,blaze::columnMajor> > dMat(dim2,dim2);

            phi.Resize(dim1,1);
            dphi.Resize(3,dim1);
            const int celDim = cel->Dimension();
            TPZManVector<REAL,3> intpoint(celDim,0.);
            REAL weight = 0.;
            STATE constitutiveParam = 0;
            material->GetDiffusiveParameter(constitutiveParam);
            REAL detjac{0};
            TPZFMatrix<REAL> jac(dim,dim),axes(dim,dim),jacinv(dim,dim);
            for(int ipt = 0; ipt < dim2; ++ipt){
                intrule->Point(ipt,intpoint,weight);
                cel->Shape(intpoint,phi,dphi);
                cel->Reference()->Jacobian(intpoint,jac,axes,detjac,jacinv);
                dMat(ipt,ipt) = weight * fabs(detjac) * constitutiveParam;
                memcpy(&bMat.data()[ipt*bMat.spacing()],phi.Adress(),bMat.spacing()*sizeof(STATE));
            }
//            blaze::DynamicMatrix<STATE> ek = bMat * dMat * trans(bMat);
            auto ek = bMat * dMat * trans(bMat);
            res+=ek(0,0);
        }
        const auto assemble_tick_e = __rdtsc();
        std::cout<<std::setw(8)<<"\tsum ek(0,0) = "<<res<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
        std::cout <<"\tTotal time: "<<((double)(assemble_tick_e-assemble_tick_b))*seconds_per_tick<<std::endl;
        std::cout <<"\tTotal MClock Cycles: "<<(assemble_tick_e-assemble_tick_b)/1e6<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
    }

    std::cout<<"Computing Element Matrices using BDBt (blaze::HybridMatrix)"<<std::endl;
    {
        STATE res{0};
        const auto assemble_tick_b = __rdtsc();
        for(const auto& iCel : cMesh->ElementVec()){
            const auto cel = dynamic_cast<TPZInterpolationSpace *>(iCel);
            if(iCel->Dimension() != dim || cel == nullptr) continue;
            TPZFNMatrix<60,STATE>phi;
            TPZFNMatrix<180,STATE>dphi;
            auto material = dynamic_cast<TPZMatMassMatrix *>(cel->Material());
            if (cel->NConnects() == 0) break;

            TPZAutoPointer<TPZIntPoints> intrule = cel->GetIntegrationRule().Clone();
            const auto n_int_points = intrule->NPoints();
            const auto dim1 = cel->NShapeF();
            const auto dim2 = n_int_points;
            //HybridMatrix<double,5UL,9UL> M7( 3UL, 7UL );
            blaze::HybridMatrix<STATE,60,150,blaze::columnMajor,blaze::unaligned,blaze::unpadded> bMat(dim1,dim2);
            blaze::DiagonalMatrix< blaze::HybridMatrix<STATE,150,150,blaze::columnMajor,blaze::unaligned,blaze::unpadded> > dMat(dim2,dim2);

            phi.Resize(dim1,1);
            dphi.Resize(3,dim1);
            const int celDim = cel->Dimension();
            TPZManVector<REAL,3> intpoint(celDim,0.);
            REAL weight = 0.;
            STATE constitutiveParam = 0;
            material->GetDiffusiveParameter(constitutiveParam);
            REAL detjac{0};
            TPZFMatrix<REAL> jac(dim,dim),axes(dim,dim),jacinv(dim,dim);
            for(int ipt = 0; ipt < dim2; ++ipt){
                intrule->Point(ipt,intpoint,weight);
                cel->Shape(intpoint,phi,dphi);
                cel->Reference()->Jacobian(intpoint,jac,axes,detjac,jacinv);
                dMat(ipt,ipt) = weight * fabs(detjac) * constitutiveParam;
                memcpy(&bMat.data()[ipt*bMat.spacing()],phi.Adress(),bMat.spacing()*sizeof(STATE));
            }
//            blaze::DynamicMatrix<STATE> ek = bMat * dMat * trans(bMat);
            auto ek = bMat * dMat * trans(bMat);
            res+=ek(0,0);
        }
        const auto assemble_tick_e = __rdtsc();
        std::cout<<std::setw(8)<<"\tsum ek(0,0) = "<<res<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
        std::cout <<"\tTotal time: "<<((double)(assemble_tick_e-assemble_tick_b))*seconds_per_tick<<std::endl;
        std::cout <<"\tTotal MClock Cycles: "<<(assemble_tick_e-assemble_tick_b)/1e6<<std::endl;
        std::cout<<"\t****************************************"<<std::endl;
    }

    delete cMesh;
    delete gMesh;
    return 0;
}

TPZCompMesh *CreateCompMesh(TPZGeoMesh *gmesh, const TPZVec<int> &matIds, const int initialPOrder, EOrthogonalFuncs familyType){
    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
    //Definition of the approximation space
    const int dim = gmesh->Dimension();
    cmesh->SetDefaultOrder(initialPOrder);
    cmesh->SetDimModel(dim);


    const int matId = matIds[0];
    constexpr REAL perm{1}, rhs{1};

    //Inserting material
    TPZMatMassMatrix * mat = new TPZMatMassMatrix(matId, dim);
    mat->SetDiffusiveParameter(perm);
    mat->SetInternalFlux(rhs);

    //Inserting volumetric materials objects
    cmesh->InsertMaterialObject(mat);

    //Boundary conditions
    constexpr int dirichlet = 0;
    constexpr int neumann = 1;
    TPZFMatrix<STATE> val1(1,1,0.0);
    TPZFMatrix<STATE> val2(1,1,0.0);
    const int &matIdBc1 = matIds[1];
    val2(0,0)=0.0;
    auto bc1 = mat->CreateBC(mat, matIdBc1, dirichlet, val1, val2);
    cmesh->InsertMaterialObject(bc1);

    cmesh->SetAllCreateFunctionsContinuous();//set H1 approximation space
    cmesh->AutoBuild();
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();

    switch(familyType){
        case EChebyshev:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Chebyshev;
            break;
        case EExpo:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Expo;
            break;
        case ELegendre:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Legendre;
            break;
        case EJacobi:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Jacobi;
            break;
        case EHermite:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Hermite;
            break;
    }

    return cmesh;
}

void FilterBoundaryEquations(TPZCompMesh *cmesh, TPZVec<int64_t> &activeEquations, int64_t &neq,
                                                 int64_t &neqOriginal) {
    neqOriginal = cmesh->NEquations();
    neq = 0;

    std::cout << "Filtering boundary equations..." << std::endl;
    TPZManVector<int64_t, 1000> allConnects;
    std::set<int64_t> boundConnects;

    for (auto iel = 0; iel < cmesh->NElements(); iel++) {
        TPZCompEl *cel = cmesh->ElementVec()[iel];
        if (cel == nullptr || cel->Reference() == nullptr) {
            continue;
        }
        TPZBndCond *mat = dynamic_cast<TPZBndCond *>(cmesh->MaterialVec()[cel->Reference()->MaterialId()]);

        //dirichlet boundary condition
        if (mat && mat->Type() == 0) {
            std::set<int64_t> boundConnectsEl;
            cel->BuildConnectList(boundConnectsEl);

            for (auto val : boundConnectsEl) {
                if (boundConnects.find(val) == boundConnects.end()) {
                    boundConnects.insert(val);
                }
            }
        }
    }

    for (auto iCon = 0; iCon < cmesh->NConnects(); iCon++) {
        if (boundConnects.find(iCon) == boundConnects.end()) {
            TPZConnect &con = cmesh->ConnectVec()[iCon];
            if(con.IsCondensed() || con.HasDependency()) continue;
            const int seqnum = con.SequenceNumber();
            if(seqnum < 0) continue;
            int pos = cmesh->Block().Position(seqnum);
            int blocksize = cmesh->Block().Size(seqnum);
            if (blocksize == 0) continue;
            int vs = activeEquations.size();
            activeEquations.Resize(vs + blocksize);
            for (int ieq = 0; ieq < blocksize; ieq++) {
                activeEquations[vs + ieq] = pos + ieq;
                neq++;
            }
        }
    }
    std::cout << "# equations(before): " << neqOriginal << std::endl;
    std::cout << "# equations(after): " << neq << std::endl;
}