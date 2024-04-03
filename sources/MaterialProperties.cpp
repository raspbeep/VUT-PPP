/**
 * @file    MaterialProperties.cpp
 * 
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *          David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains class which represent materials in the simulation
 *          domain.
 *
 * @date    2024-02-22
 */

#include <cstddef>
#include <ios>
#include <string_view>
#include <vector>

#include <hdf5.h>

#include "Hdf5Handle.hpp"
#include "MaterialProperties.hpp"

void MaterialProperties::load(std::string_view fileName, bool loadData)
{
  // 1. Open the input file for reading only.
  Hdf5FileHandle fileHandle(H5Fopen(fileName.data(), H5F_ACC_RDONLY, H5P_DEFAULT));
  
  if (!fileHandle.valid())
  {
    throw std::ios::failure("Cannot open input file");
  }

  {
    // 2. Read "/EdgeSize" dataset which is scalar and contains size of the
    //    domain.
    Hdf5DatasetHandle datasetHandle(H5Dopen(fileHandle, "/EdgeSize", H5P_DEFAULT));

    if (!datasetHandle.valid())
    {
      throw std::ios::failure("Cannot open dataset EdgeSize");
    }

    H5Dread(datasetHandle, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mEdgeSize);
  }

  mGridPointCount = mEdgeSize * mEdgeSize;

  {
    // 3. Read "/CoolerTemp" dataset which is scalar containing temperature
    //    of the cooler.
    Hdf5DatasetHandle datasetHandle(H5Dopen(fileHandle, "/CoolerTemp", H5P_DEFAULT));

    if (!datasetHandle.valid())
    {
      throw std::ios::failure("Cannot open dataset CoolerTemp");
    }

    H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mCoolerTemp);
  }

  {
    // 4. Read "/HeaterTemp" dataset which is scalar containing temperature
    //    of the heater element (CPU).
    Hdf5DatasetHandle datasetHandle(H5Dopen(fileHandle, "/HeaterTemp", H5P_DEFAULT));

    if (!datasetHandle.valid())
    {
      throw std::ios::failure("Cannot open dataset HeaterTemp");
    }
    
    H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mHeaterTemp);
  }

  // Do we want to actually read the data?
  if(loadData)
  {
    // Allocate memory for data in the domain.
    mDomainMap.resize(mGridPointCount);
    mDomainParams.resize(mGridPointCount);
    mInitTemp.resize(mGridPointCount);

    {
      // 5. Read "/DomainMap" which contains material specifications.
      Hdf5DatasetHandle datasetHandle(H5Dopen(fileHandle, "/DomainMap", H5P_DEFAULT));

      if (!datasetHandle.valid())
      {
        throw std::ios::failure("Cannot open dataset DomainMap");
      }

      H5Dread(datasetHandle, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mDomainMap.data());
    }

    {
      // 6. Read "/DomainParameters" which contains thermal properties of the material.
      Hdf5DatasetHandle datasetHandle(H5Dopen(fileHandle, "/DomainParameters", H5P_DEFAULT));

      if (!datasetHandle.valid())
      {
        throw std::ios::failure("Cannot open dataset DomainParameters");
      }

      H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mDomainParams.data());
    }

    {
      // 7. Read "/InitialTemperature" which contains initial temperature distribution.
      Hdf5DatasetHandle datasetHandle(H5Dopen(fileHandle, "/InitialTemperature", H5P_DEFAULT));

      if (!datasetHandle.valid())
      {
        throw std::ios::failure("Cannot open dataset InitialTemperature");
      }

      H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mInitTemp.data());
    }
  }
}

const std::vector<int, AlignedAllocator<int>>& MaterialProperties::getDomainMap() const
{
  return mDomainMap;
}

const std::vector<float, AlignedAllocator<float>>& MaterialProperties::getDomainParameters() const
{
  return mDomainParams;
}

const std::vector<float, AlignedAllocator<float>>& MaterialProperties::getInitialTemperature() const
{
  return mInitTemp;
}

float MaterialProperties::getCoolerTemperature() const
{
  return mCoolerTemp;
}

float MaterialProperties::getHeaterTemperature() const
{
  return mHeaterTemp;
}

std::size_t MaterialProperties::getEdgeSize() const
{
  return mEdgeSize;
}

std::size_t MaterialProperties::getGridPointCount() const
{
  return mGridPointCount;
}

