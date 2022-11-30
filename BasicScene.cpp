#include "BasicScene.h"
#include <read_triangle_mesh.h>
#include <utility>
#include <min_heap.h>
#include "ObjLoader.h"
#include "IglMeshLoader.h"
#include "igl/read_triangle_mesh.cpp"
#include "igl/edge_flaps.h"
#include <igl/circulation.h>
#include <igl/collapse_edge.h>
#include <igl/edge_flaps.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/parallel_for.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <iostream>
#include <set>
#include <per_vertex_normals.h>
// #include "AutoMorphingModel.h"

using namespace cg3d;
std::shared_ptr<Movable> root;
std::shared_ptr<cg3d::Model> cyl, sphere1 ,cube;
Eigen::VectorXi EMAP;
Eigen::MatrixXi F,E,EF,EI;
Eigen::VectorXi EQ;
// If an edge were collapsed, we'd collapse it to these points:
Eigen::MatrixXd V, C, N,T;
Eigen::MatrixXd OV;
Eigen::MatrixXi OF;
igl::min_heap< std::tuple<double,int,int> > Q;
int num_collapsed;

void BasicScene::decrease() {
    reset();
    if(!Q.empty()) {
        bool something_collapsed = false;
        // collapse edge
        const int max_iter = std::ceil(0.1 * Q.size());
        for (int j = 0; j < max_iter; j++) {
            if (!igl::collapse_edge(igl::shortest_edge_and_midpoint, V, F, E, EMAP, EF, EI, Q, EQ, C)) {
                break;
            }
            something_collapsed = true;
            num_collapsed++;
        }

        if (something_collapsed) {
            igl::per_vertex_normals(V, F, N);
            T = Eigen::MatrixXd::Zero(V.rows(), 2);
            std::vector<cg3d::MeshData> newMeshData;
            newMeshData.push_back({V, F, N, T});

            std::shared_ptr<cg3d::Mesh> newMesh = std::make_shared<cg3d::Mesh>("new mesh", newMeshData);

//                        auto mesh=cyl->GetMeshList();
//                        mesh[0]->data.push_back({V,F,N,T});
            cyl->GetMeshList().insert(cyl->GetMeshList().begin(), newMesh);
            cyl->meshIndex = 1;
        }
    }


}

void BasicScene::reset() {
    num_collapsed=0;

    auto mesh = cyl->GetMeshList();
    V=mesh[0]->data[0].vertices;
    F = mesh[0]->data[0].faces;
    igl::edge_flaps(F, E, EMAP, EF, EI);
    C.resize(E.rows(), V.cols());
    Eigen::VectorXd costs(E.rows());
    // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
    // Q.clear();
    Q = {};
    EQ = Eigen::VectorXi::Zero(E.rows());
    {
        Eigen::VectorXd costs(E.rows());
        igl::parallel_for(E.rows(), [&](const int e) {
            double cost = e;
            Eigen::RowVectorXd p(1, 3);
            igl::shortest_edge_and_midpoint(e, V, F, E, EMAP, EF, EI, cost, p);
            C.row(e) = p;
            costs(e) = cost;
        }, 10000);
        for (int e = 0; e < E.rows(); e++) {
            Q.emplace(costs(e), e, 0);
        }
    }

}

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    camera = Camera::Create( "camera", fov, float(width) / height, near, far);

    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    auto daylight{std::make_shared<Material>("daylight", "shaders/cubemapShader")};
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{Model::Create("background", Mesh::Cube(), daylight)};
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();


    auto program = std::make_shared<Program>("shaders/basicShader");
    auto material{ std::make_shared<Material>("material", program)}; // empty material
//    SetNamedObject(cube, Model::Create, Mesh::Cube(), material, shared_from_this());

    material->AddTexture(0, "textures/box0.bmp", 2);
  //  auto sphereMesh{IglLoader::MeshFromFiles("sphere_igl", "data/sphere.obj")};
    auto cylMesh{IglLoader::MeshFromFiles("cyl_igl","data/camel_b.obj")};
   // auto cubeMesh{IglLoader::MeshFromFiles("cube_igl","data/cube.off")};

   // sphere1 = Model::Create( "sphere",sphereMesh, material);
    cyl = Model::Create( "cyl", cylMesh, material);
  //  cube = Model::Create( "cube", cubeMesh, material);
//    sphere1->Scale(2);
//    sphere1->showWireframe = true;
//    sphere1->Translate({-3,0,0});
    cyl->Translate({3,0,0});
    cyl->Scale(0.12f);
    cyl->showWireframe = true;
//    cube->showWireframe = true;
    camera->Translate(20, Axis::Z);
   // root->AddChild(sphere1);
    root->AddChild(cyl);
  //  root->AddChild(cube);


    //igl::read_triangle_mesh("data/cube.off",V,F);
    //igl::edge_flaps(F,E,EMAP,EF,EI);
    std::cout<< "vertices: \n" << V <<std::endl;
    std::cout<< "faces: \n" << F <<std::endl;

    std::cout<< "edges: \n" << E.transpose() <<std::endl;
    std::cout<< "edges to faces: \n" << EF.transpose() <<std::endl;
    std::cout<< "faces to edges: \n "<< EMAP.transpose()<<std::endl;
    std::cout<< "edges indices: \n" << EI.transpose() <<std::endl;


    reset();
    auto mesh = cyl->GetMeshList();
    igl::per_vertex_normals(V,F,N);
    T= Eigen::MatrixXd::Zero(V.rows(),2);
    //mesh=pickedModel->GetMeshList();
    mesh[0]->data.push_back({V,F,N,T});
    cyl->SetMeshList(mesh);
    cyl->meshIndex = 1;
}

    void BasicScene::Update(const Program &program, const Eigen::Matrix4f &proj, const Eigen::Matrix4f &view,
                            const Eigen::Matrix4f &model) {
        Scene::Update(program, proj, view, model);
        program.SetUniform4f("lightColor", 1.0f, 1.0f, 1.0f, 0.5f);
        program.SetUniform4f("Kai", 1.0f, 1.0f, 1.0f, 1.0f);
        //cube->Rotate(0.01f, Axis::All);
    }

    void BasicScene::KeyCallback(cg3d::Viewport *_viewport, int x, int y, int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            switch (key) // NOLINT(hicpp-multiway-paths-covered)
            {
                case GLFW_KEY_SPACE:
                    decrease();


            }

        }
    }





