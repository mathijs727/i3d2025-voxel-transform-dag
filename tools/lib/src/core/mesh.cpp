#include "voxcom/core/mesh.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <cstring> // missing from assimp/mesh.h
#include <exception>
#include <memory_resource>
#include <numeric>
#include <span>
#include <stack>
#include <string>
#include <tuple>
#include <voxcom/format/fmt_filesystem.h>
#include <voxcom/format/fmt_glm.h>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

namespace voxcom {

static glm::mat4 assimpMatrix(const aiMatrix4x4& m)
{
    // float values[3][4] = {};
    glm::mat4 matrix;
    matrix[0][0] = m.a1;
    matrix[0][1] = m.b1;
    matrix[0][2] = m.c1;
    matrix[0][3] = m.d1;
    matrix[1][0] = m.a2;
    matrix[1][1] = m.b2;
    matrix[1][2] = m.c2;
    matrix[1][3] = m.d2;
    matrix[2][0] = m.a3;
    matrix[2][1] = m.b3;
    matrix[2][2] = m.c3;
    matrix[2][3] = m.d3;
    matrix[3][0] = m.a4;
    matrix[3][1] = m.b4;
    matrix[3][2] = m.c4;
    matrix[3][3] = m.d4;
    return matrix;
}

static glm::vec3 assimpVec(const aiVector3D& v)
{
    return glm::vec3(v.x, v.y, v.z);
}

static glm::vec3 assimpVec(const aiColor3D& c)
{
    return glm::vec3(c.r, c.g, c.b);
}

using TextureCache = std::unordered_map<std::string, std::shared_ptr<Image2D<RGB>>>;

static std::shared_ptr<Image2D<RGB>> loadMaterialTexture(const aiMaterial* pAssimpMaterial, aiTextureType textureType, const aiScene* pAssimpScene, const std::filesystem::path& textureBasePath, TextureCache& textureCache)
{
    aiString relativeTexturePath;
    if (pAssimpMaterial->GetTexture(textureType, 0, &relativeTexturePath) == AI_SUCCESS) {
        const std::string relativeTexturePathStr { relativeTexturePath.C_Str() };
        if (auto iter = textureCache.find(relativeTexturePathStr); iter != std::end(textureCache))
            return iter->second;

        std::shared_ptr<Image2D<RGB>> out;
        if (auto pTexture = pAssimpScene->GetEmbeddedTexture(relativeTexturePath.C_Str())) {
            out = std::make_shared<Image2D<voxcom::RGB>>(pTexture);
        } else {
            // Scene/mesh file refers to an external texture.
            std::filesystem::path absoluteTexturePath = textureBasePath / relativeTexturePathStr;
            if (std::filesystem::exists(absoluteTexturePath))
                out = std::make_shared<Image2D<voxcom::RGB>>(absoluteTexturePath);
            else
                spdlog::error("Cannot find texture {}", absoluteTexturePath.string());
        }

        textureCache[relativeTexturePathStr] = out;
        return out;
    }

    return nullptr;
}

std::vector<Mesh> loadMeshes(const std::filesystem::path& filePath)
{
    if (!std::filesystem::exists(filePath)) {
        spdlog::error("File \"{}\" does not exist", filePath.string());
        throw std::exception();
    }

    std::filesystem::path textureBasePath = std::filesystem::absolute(filePath).parent_path();
    TextureCache textureCache;
    Assimp::Importer importer;
    const aiScene* pAssimpScene = importer.ReadFile(filePath.string().c_str(), aiProcess_Triangulate);

    if (pAssimpScene == nullptr || pAssimpScene->mRootNode == nullptr || pAssimpScene->mFlags == AI_SCENE_FLAGS_INCOMPLETE) {
        spdlog::error("Assimp failed to load mesh \"{}\"", filePath.string());
        throw std::exception();
    }

    std::vector<Mesh> out;
    std::stack<std::tuple<aiNode*, glm::mat4>> stack;
    stack.push({ pAssimpScene->mRootNode, assimpMatrix(pAssimpScene->mRootNode->mTransformation) });
    while (!stack.empty()) {
        auto [pNode, matrix] = stack.top();
        stack.pop();

        matrix *= assimpMatrix(pNode->mTransformation);
        [[maybe_unused]] const glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(matrix));

        for (unsigned i = 0; i < pNode->mNumMeshes; i++) {
            // Process sub mesh.
            const aiMesh* pAssimpMesh = pAssimpScene->mMeshes[pNode->mMeshes[i]];

            if (pAssimpMesh->mNumVertices == 0 || pAssimpMesh->mNumFaces == 0)
                spdlog::warn("Empty mesh encountered");

            // Process triangles in sub mesh.
            Mesh mesh;
            for (unsigned j = 0; j < pAssimpMesh->mNumFaces; j++) {
                const aiFace& face = pAssimpMesh->mFaces[j];
                if (face.mNumIndices != 3) {
                    spdlog::warn("Found a face which is not a triangle, discarding!");
                    continue;
                }

                const auto aiIndices = face.mIndices;
                mesh.triangles.emplace_back(aiIndices[0], aiIndices[1], aiIndices[2]);
            }

            // Process vertices in sub mesh.
            for (unsigned j = 0; j < pAssimpMesh->mNumVertices; j++) {
                mesh.positions.push_back(matrix * glm::vec4(assimpVec(pAssimpMesh->mVertices[j]), 1.0f));
                if (pAssimpMesh->HasTextureCoords(0)) {
                    auto tmp = assimpVec(pAssimpMesh->mTextureCoords[0][j]);
                    tmp.y = 1.0f - tmp.y;
                    mesh.texCoords.push_back(tmp);
                }
                if (pAssimpMesh->HasTextureCoords(1)) {
                    auto tmp = assimpVec(pAssimpMesh->mTextureCoords[1][j]);
                    tmp.y = 1.0f - tmp.y;
                    mesh.lightMapCoords.push_back(tmp);
                }
            }

            // Read the material, more info can be found here:
            // http://assimp.sourceforge.net/lib_html/materials.html
            const aiMaterial* pAssimpMaterial = pAssimpScene->mMaterials[pAssimpMesh->mMaterialIndex];
            auto getMaterialColor = [&](const char* pKey, unsigned type, unsigned idx) {
                aiColor3D color { 0.0f, 0.0f, 0.0f };
                pAssimpMaterial->Get(pKey, type, idx, color);
                return assimpVec(color);
            };
            [[maybe_unused]] auto getMaterialFloat = [&](const char* pKey, unsigned type, unsigned idx) {
                float value;
                pAssimpMaterial->Get(pKey, type, idx, value);
                return value;
            };

            mesh.pDiffuseTexture = loadMaterialTexture(pAssimpMaterial, aiTextureType_DIFFUSE, pAssimpScene, textureBasePath, textureCache);
            mesh.pLightMapTexture = loadMaterialTexture(pAssimpMaterial, aiTextureType_LIGHTMAP, pAssimpScene, textureBasePath, textureCache);
            // mesh.pLightMapTexture = tryLoadUnityLightmap(pAssimpScene, textureBasePath, pNode->mMetaData);
            const glm::vec3 color01 = getMaterialColor(AI_MATKEY_COLOR_DIFFUSE);
            mesh.diffuseBaseColor = { uint8_t(color01.r * 255.0f), uint8_t(color01.g * 255.0f), uint8_t(color01.b * 255.0f) };

            out.emplace_back(std::move(mesh));
        }

        for (unsigned i = 0; i < pNode->mNumChildren; i++) {
            stack.push({ pNode->mChildren[i], matrix });
        }
    }
    importer.FreeScene();

    return out;
}

void saveMesh(std::span<const Mesh> meshes, const std::filesystem::path& filePath)
{
    aiScene scene;
    scene.mRootNode = new aiNode();

    scene.mMaterials = new aiMaterial*[meshes.size()];
    scene.mNumMaterials = (unsigned)meshes.size();

    scene.mMeshes = new aiMesh*[meshes.size()];
    scene.mNumMeshes = (unsigned)meshes.size();
    scene.mRootNode->mMeshes = new unsigned int[meshes.size()];
    for (unsigned i = 0; i < meshes.size(); i++)
        scene.mRootNode->mMeshes[i] = i;
    scene.mRootNode->mNumMeshes = (unsigned)meshes.size();

    for (unsigned meshIdx = 0; meshIdx < meshes.size(); meshIdx++) {
        const auto& mesh = meshes[meshIdx];

        auto*& pMesh = scene.mMeshes[meshIdx];
        pMesh = new aiMesh();
        pMesh->mMaterialIndex = meshIdx;
        auto* pMaterial = scene.mMaterials[meshIdx] = new aiMaterial();
        const aiColor3D color { mesh.diffuseBaseColor.r / 255.0f, mesh.diffuseBaseColor.g / 255.0f, mesh.diffuseBaseColor.b / 255.0f };
        pMaterial->AddProperty(&color, 1, AI_MATKEY_COLOR_DIFFUSE);
        pMesh->mVertices = new aiVector3D[mesh.positions.size()];
        pMesh->mNumVertices = (unsigned)mesh.positions.size();
        pMesh->mNumUVComponents[0] = 0;

        for (unsigned i = 0; i < mesh.positions.size(); i++) {
            const auto& v = mesh.positions[i];
            pMesh->mVertices[i] = aiVector3D(v.x, v.y, v.z);
        }

        if (!mesh.quads.empty()) {
            pMesh->mPrimitiveTypes = aiPrimitiveType_POLYGON;
            pMesh->mFaces = new aiFace[mesh.quads.size()];
            pMesh->mNumFaces = (unsigned)mesh.quads.size();
            pMesh->mNumFaces = (unsigned)mesh.quads.size();
            pMesh->mNormals = new aiVector3D[mesh.positions.size()];
            for (unsigned i = 0; i < mesh.quads.size(); i++) {
                aiFace& face = pMesh->mFaces[i];

                const auto v0 = mesh.positions[mesh.quads[i][0]];
                const auto v1 = mesh.positions[mesh.quads[i][0]];
                const auto v2 = mesh.positions[mesh.quads[i][0]];
                const auto n = glm::normalize(glm::cross(v2 - v0, v1 - v0));
                pMesh->mNormals[mesh.quads[i][0]] = aiVector3D(n.x, n.y, n.z);
                pMesh->mNormals[mesh.quads[i][1]] = aiVector3D(n.x, n.y, n.z);
                pMesh->mNormals[mesh.quads[i][2]] = aiVector3D(n.x, n.y, n.z);
                pMesh->mNormals[mesh.quads[i][3]] = aiVector3D(n.x, n.y, n.z);

                face.mIndices = new unsigned int[4];
                face.mNumIndices = 4;
                face.mIndices[0] = mesh.quads[i][0];
                face.mIndices[1] = mesh.quads[i][1];
                face.mIndices[2] = mesh.quads[i][2];
                face.mIndices[3] = mesh.quads[i][3];
            }
        } else {
            pMesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
            pMesh->mFaces = new aiFace[mesh.triangles.size()];
            pMesh->mNumFaces = (unsigned)mesh.triangles.size();
            for (unsigned i = 0; i < mesh.triangles.size(); i++) {
                aiFace& face = pMesh->mFaces[i];

                face.mIndices = new unsigned int[3];
                face.mNumIndices = 3;

                face.mIndices[0] = mesh.triangles[i][0];
                face.mIndices[1] = mesh.triangles[i][1];
                face.mIndices[2] = mesh.triangles[i][2];
            }
        }
    }

    Assimp::Exporter exporter {};
    const auto format = filePath.extension().string().substr(1);
    exporter.Export(&scene, format.c_str(), filePath.string().c_str());
}

void saveMesh(const Mesh& mesh, const std::filesystem::path& filePath)
{
    saveMesh(std::span(&mesh, 1), filePath);
}

Bounds Mesh::computeBounds() const
{
    Bounds out;
    for (const auto& pos : positions)
        out.grow(pos);
    return out;
}
}