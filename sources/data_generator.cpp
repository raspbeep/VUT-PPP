/**
 * @file        data_generator.cpp
 * 
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *             
 * @author      David Bayer \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              ibayer@fit.vutbr.cz
 *
 * @brief       The implementation file creating a file with all the material
 *              properties of the domain.
 *
 * @version     2024
 * @date        19 February 2015, 16:22 (created) \n
 *              01 March 2022, 14:52 (revised) \n
 *              22 February 2024, 14:52 (revised)
 *
 * @details     This is the data generator for PPP 2024 projects
 */

#include <cstdlib>
#include <memory>
#include <string>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <cxxopts.hpp>

//----------------------------------------------------------------------------//
//------------------------- Data types declarations --------------------------//
//----------------------------------------------------------------------------//

/**
 * @struct TParameters
 * @brief  Parameters of the program
 */
struct TParameters
{
  std::string FileName;
  std::size_t Size;
  float       HeaterTemperature;
  float       CoolerTemperature;

  float       dt;
  float       dx;
};// end of Parameters
//------------------------------------------------------------------------------

/**
 * @struct MediumParameters
 * @brief Parameters of Medium
 */
struct TMediumParameters
{
  float k_s;     // W/(m K)  Thermal conductivity - conduction ciefficient
  float rho;     // kg.m^3   Density
  float Cp;      // J/kg K   Spefic heat constant pressure
  float alpha;   // m^2/s    Diffusivity

  TMediumParameters(const float k_s, const float rho, const float Cp)
                   : k_s(k_s), rho(rho), Cp(Cp)
  {
    alpha = k_s / (rho * Cp);
  }

  // Calculate coef F0 - heat diffusion parameter
  inline float GetF0(const float dx, const float dt) const
  {
    return alpha * dt / (dx * dx) ;
  }

  /// Check stability of the simulation for the medium
  inline bool CheckStability(const float dx, const float dt) const
  {
    return (GetF0(dx, dt) < 0.25f);
  }
};// end of TMediumParameters
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//-------------------------    Global variables        -----------------------//
//----------------------------------------------------------------------------//

constexpr std::size_t MaskSize       =  16;  // size of the mask
constexpr float       RealDomainSize =  1.f; // length of the edge 1m


///  Basic mask of the cooler (0 - Air, 1 -aluminum, 2 - copper)
int CoolerMask[MaskSize * MaskSize]
//1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
{ 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0,   //16
  0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0,   //15
  0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,   //14
  0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0,   //13
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   //12
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   //11
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   //10
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   // 9
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   // 8
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   // 7
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   // 6
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   // 5
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   // 4
  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,   // 3
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   // 2
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    // 1
};


/// Parameters of the medium
TParameters Parameters{};

/// Properties of Air
TMediumParameters Air(0.0024f, 1.207f, 1006.1f);

/// Properties of Aluminum
TMediumParameters Aluminum(205.f, 2700.f, 910.f);
//Aluminum.SetValues()

/// Properties of Copper
TMediumParameters Copper(387.f, 8940.f, 380.f);



//----------------------------------------------------------------------------//
//------------------------- Function declarations ----------------------------//
//----------------------------------------------------------------------------//

/// Set parameters
void ParseCommandline(int argc, char** argv);

/// Generate data for the matrix
void GenerateData(int DomainMap[], float DomainParameters[]);

/// Store data in the file
void StoreData();

// Get dx
float Getdx();

//----------------------------------------------------------------------------//
//------------------------- Function implementation  -------------------------//
//----------------------------------------------------------------------------//
  
/**
 * Parse commandline and setup
 * @param [in] argc
 * @param [in] argv
 */
void ParseCommandline(int argc, char** argv)
{
  cxxopts::Options options("data_generator", "creating a material properties file of the domain");

  options.add_options()
    ("o,output", "Output file name with the medium data",
     cxxopts::value<std::string>()->default_value("ppp_input_data.h5"), "<string>")
    ("n,size", "Size of the domain (power of 2 only)", cxxopts::value<std::size_t>()->default_value("16"), "<uint>")
    ("H,heater-temperature", "Heater temperature °C", cxxopts::value<float>()->default_value("100.f"), "<float>")
    ("C,air-temperature", "Cooler temperature °C", cxxopts::value<float>()->default_value("20.f"), "<float>")
    ("h,help", "Print usage");

  try
  {
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::printf("%s", options.help().c_str());
      std::exit(EXIT_SUCCESS);
    }

    Parameters.Size              = result["size"].as<std::size_t>();
    Parameters.FileName          = result["output"].as<std::string>();
    Parameters.HeaterTemperature = result["heater-temperature"].as<float>();
    Parameters.CoolerTemperature = result["air-temperature"].as<float>();

    if (!((Parameters.Size != 0) && !(Parameters.Size & (Parameters.Size - 1))))
    {
      throw std::runtime_error("The size is not power of two");
    }

    if (Parameters.Size < 16)
    {
      throw std::runtime_error("Minimum size is 16");
    }
  }
  catch (const std::exception& e)
  {
    std::printf("Error: %s\n\n", e.what());
    std::printf("%s", options.help().c_str());
    std::exit(EXIT_FAILURE);
  }

  if (Parameters.Size < 128)        Parameters.dt = 0.1f;
  else if (Parameters.Size < 512)   Parameters.dt = 0.01f;
  else if (Parameters.Size < 2048)  Parameters.dt = 0.001f;
  else if (Parameters.Size < 16384) Parameters.dt = 0.0001f;
  else                              Parameters.dt = 0.00001f;

  Parameters.dx = RealDomainSize / static_cast<float>(Parameters.Size);
}// end of ParseCommandline
//------------------------------------------------------------------------------


/**
 * Generate data for the domain
 * @param [out] DomainMap
 * @param [out] DomainParameters
 * @param [out] InitialTemperature
 */
void GenerateData(int * DomainMap, float * DomainParameters, float * InitialTemperature)
{
  const std::size_t ScaleFactor = Parameters.Size / MaskSize;

  // set the global medium map
  #pragma omp parallel
  {
    #pragma omp for
    for (std::size_t m_y = 0; m_y < MaskSize; m_y++)
    {
      for (std::size_t m_x = 0; m_x < MaskSize; m_x++)
      {
        // Scale
        for (std::size_t y = 0; y < ScaleFactor; y++)
        {
          for (std::size_t x = 0; x < ScaleFactor; x++)
          {
            std::size_t global = (m_y * ScaleFactor + y)* Parameters.Size + (m_x * ScaleFactor + x);
            std::size_t local = m_y * MaskSize + m_x;

            DomainMap[global]  = CoolerMask[local];
            //
          }// x
        }// y
      } // m_x
    }// m_y

    // set medium properties
    #pragma omp for
    for (std::size_t y = 0; y < Parameters.Size; y++)
    {
      for (std::size_t x = 0; x < Parameters.Size; x++)
      {
        switch(DomainMap[y * Parameters.Size + x])
        {
          case 0: DomainParameters[y * Parameters.Size + x] = Air.GetF0(Parameters.dx, Parameters.dt); break;
          case 1: DomainParameters[y * Parameters.Size + x] = Aluminum.GetF0(Parameters.dx, Parameters.dt); break;
          case 2: DomainParameters[y * Parameters.Size + x] = Copper.GetF0(Parameters.dx, Parameters.dt); break;
        }
      }
    }

    // set initial temperature (skip first two lines)  - that's the heater
    #pragma omp for
    for (std::size_t y = 2; y < Parameters.Size; y++)
    {
      for (std::size_t x = 0; x < Parameters.Size; x++)
      {
        InitialTemperature[y * Parameters.Size + x] = Parameters.CoolerTemperature;
      }
    }
  }// end of parallel

  //set temperature for heater
  for (std::size_t x = 0; x < 2*Parameters.Size; x++)
  { // where is cooper, set Heater
    InitialTemperature[x] = (DomainMap[x] == 2) ? Parameters.HeaterTemperature : Parameters.CoolerTemperature;
  }
}// end of GenerateData
//------------------------------------------------------------------------------


/**
 * Store data in the file
 * @param [in] DomainMap
 * @param [in] DomainParameters
 * @param [in] InitialTemperature
 */
void StoreData(const int * DomainMap, const float * DomainParameters, const float * InitialTemperature)
{
  hid_t HDF5_File = H5Fcreate(Parameters.FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t ScalarSize[1] {1};
  hsize_t DomainSize[2] {Parameters.Size,Parameters.Size};

  long Size = static_cast<long>(Parameters.Size);

  H5LTmake_dataset_long(HDF5_File,  "/EdgeSize",           1, ScalarSize, &Size);
  H5LTmake_dataset_float(HDF5_File, "/CoolerTemp",         1, ScalarSize, &Parameters.CoolerTemperature);
  H5LTmake_dataset_float(HDF5_File, "/HeaterTemp",         1, ScalarSize, &Parameters.HeaterTemperature);
  H5LTmake_dataset_int (HDF5_File,  "/DomainMap",          2, DomainSize, DomainMap);
  H5LTmake_dataset_float(HDF5_File, "/DomainParameters",   2, DomainSize, DomainParameters);
  H5LTmake_dataset_float(HDF5_File, "/InitialTemperature", 2, DomainSize, InitialTemperature);

  H5Fclose(HDF5_File);
}// end of StoreData
//------------------------------------------------------------------------------

/**
 * main function
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char** argv)
{
  ParseCommandline(argc,argv);

  std::printf("---------------------------------------------\n");
  std::printf("--------- PPP 2020 data generator -----------\n");
  std::printf("File name  : %s\n",   Parameters.FileName.c_str());
  std::printf("Size       : [%zu,%zu]\n",   Parameters.Size, Parameters.Size);
  std::printf("Heater temp: %.2fC\n", Parameters.HeaterTemperature);
  std::printf("Cooler temp: %.2f\n", Parameters.CoolerTemperature);

  auto DomainMap          = std::make_unique<int[]>(Parameters.Size * Parameters.Size);
  auto DomainParameters   = std::make_unique<float[]>(Parameters.Size * Parameters.Size);
  auto InitialTemperature = std::make_unique<float[]>(Parameters.Size * Parameters.Size);

  std::printf("Air      : %f\n", Air.GetF0(Parameters.dx,Parameters.dt));
  std::printf("Aluminum : %f\n", Aluminum.GetF0(Parameters.dx,Parameters.dt));
  std::printf("Copper   : %f\n", Copper.GetF0(Parameters.dx,Parameters.dt));

  if (!(Copper.CheckStability(Parameters.dx,Parameters.dt) &&
      Aluminum.CheckStability(Parameters.dx,Parameters.dt) &&
      Air.CheckStability(Parameters.dx,Parameters.dt)))
  {
    std::printf("dt and dx are too big, simulation may be unstable! \n");
  }

  std::printf("Generating data ...");
  GenerateData(DomainMap.get(), DomainParameters.get(), InitialTemperature.get());
  std::printf("Done\n");

  std::printf("Storing data ...");
  StoreData(DomainMap.get(), DomainParameters.get(), InitialTemperature.get());
  std::printf("Done\n");
}// end of main
//------------------------------------------------------------------------------
