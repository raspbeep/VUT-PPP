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

#include <memory>
#include <string>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <cxxopts.hpp>
#include <fmt/format.h>

//----------------------------------------------------------------------------//
//------------------------- Data types declarations --------------------------//
//----------------------------------------------------------------------------//

/**
 * @struct TParameters
 * @brief  Parameters of the program
 */
struct TParameters
{
  std::string fileName{};
  std::size_t size{};
  float       heaterTemperature{};
  float       coolerTemperature{};

  float       dt{};
  float       dx{};
};// end of Parameters
//------------------------------------------------------------------------------

/**
 * @struct MediumParameters
 * @brief Parameters of Medium
 */
struct TMediumParameters
{
  float k_s{};     // W/(m K)  Thermal conductivity - conduction ciefficient
  float rho{};     // kg.m^3   Density
  float Cp{};      // J/kg K   Spefic heat constant pressure
  float alpha{};   // m^2/s    Diffusivity

  TMediumParameters(const float k_s, const float rho, const float Cp)
  : k_s(k_s), rho(rho), Cp(Cp)
  {
    alpha = k_s / (rho * Cp);
  }

  // Calculate coef F0 - heat diffusion parameter
  float getF0(const float dx, const float dt) const
  {
    return alpha * dt / (dx * dx) ;
  }

  /// Check stability of the simulation for the medium
  bool checkStability(const float dx, const float dt) const
  {
    return (getF0(dx, dt) < 0.25f);
  }
};// end of TMediumParameters
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//-------------------------    Global variables        -----------------------//
//----------------------------------------------------------------------------//

constexpr std::size_t maskSize       =  16;  // size of the mask
constexpr float       realDomainSize =  1.f; // length of the edge 1m


///  Basic mask of the cooler (0 - Air, 1 -aluminum, 2 - copper)
int coolerMask[maskSize * maskSize]
{
//1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
  0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0,   //16
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
TParameters parameters{};

/// Properties of Air
TMediumParameters air(0.0024f, 1.207f, 1006.1f);

/// Properties of Aluminum
TMediumParameters aluminum(205.f, 2700.f, 910.f);
//Aluminum.SetValues()

/// Properties of Copper
TMediumParameters copper(387.f, 8940.f, 380.f);



//----------------------------------------------------------------------------//
//------------------------- Function declarations ----------------------------//
//----------------------------------------------------------------------------//

/// Set parameters
void parseCommandline(int argc, char** argv);

/// Generate data for the matrix
void generateData(int DomainMap[], float DomainParameters[]);

/// Store data in the file
void storeData();

//----------------------------------------------------------------------------//
//------------------------- Function implementation  -------------------------//
//----------------------------------------------------------------------------//
  
/**
 * Parse commandline and setup
 * @param [in] argc
 * @param [in] argv
 */
void parseCommandline(int argc, char** argv)
{
  cxxopts::Options options("data_generator", "Program for creating a material properties file of the domain");

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
      fmt::print("{}\n", options.help());
      std::exit(EXIT_SUCCESS);
    }

    parameters.size              = result["size"].as<std::size_t>();
    parameters.fileName          = result["output"].as<std::string>();
    parameters.heaterTemperature = result["heater-temperature"].as<float>();
    parameters.coolerTemperature = result["air-temperature"].as<float>();

    if (!((parameters.size != 0) && !(parameters.size & (parameters.size - 1))))
    {
      throw std::runtime_error("The size is not power of two");
    }

    if (parameters.size < 16)
    {
      throw std::runtime_error("Minimum size is 16");
    }
  }
  catch (const std::exception& e)
  {
    fmt::print(stderr, "Error: {}\n\n", e.what());
    fmt::print(stderr, "{}\n", options.help());
    std::exit(EXIT_FAILURE);
  }

  if (parameters.size < 128)        parameters.dt = 0.1f;
  else if (parameters.size < 512)   parameters.dt = 0.01f;
  else if (parameters.size < 2048)  parameters.dt = 0.001f;
  else if (parameters.size < 16384) parameters.dt = 0.0001f;
  else                              parameters.dt = 0.00001f;

  parameters.dx = realDomainSize / static_cast<float>(parameters.size);
}// end of parseCommandline
//------------------------------------------------------------------------------


/**
 * Generate data for the domain
 * @param [out] DomainMap
 * @param [out] DomainParameters
 * @param [out] InitialTemperature
 */
void generateData(int* domainMap, float* domainParameters, float* initialTemperature)
{
  const std::size_t scaleFactor = parameters.size / maskSize;

  // set the global medium map
# pragma omp parallel
  {
#   pragma omp for
    for (std::size_t m_y = 0; m_y < maskSize; m_y++)
    {
      for (std::size_t m_x = 0; m_x < maskSize; m_x++)
      {
        // Scale
        for (std::size_t y = 0; y < scaleFactor; y++)
        {
          for (std::size_t x = 0; x < scaleFactor; x++)
          {
            std::size_t global = (m_y * scaleFactor + y)* parameters.size + (m_x * scaleFactor + x);
            std::size_t local = m_y * maskSize + m_x;

            domainMap[global]  = coolerMask[local];
            //
          }// x
        }// y
      } // m_x
    }// m_y

    // set medium properties
#   pragma omp for
    for (std::size_t y = 0; y < parameters.size; y++)
    {
      for (std::size_t x = 0; x < parameters.size; x++)
      {
        switch(domainMap[y * parameters.size + x])
        {
          case 0: domainParameters[y * parameters.size + x] = air.getF0(parameters.dx, parameters.dt); break;
          case 1: domainParameters[y * parameters.size + x] = aluminum.getF0(parameters.dx, parameters.dt); break;
          case 2: domainParameters[y * parameters.size + x] = copper.getF0(parameters.dx, parameters.dt); break;
        }
      }
    }

    // set initial temperature (skip first two lines)  - that's the heater
#   pragma omp for
    for (std::size_t y = 2; y < parameters.size; y++)
    {
      for (std::size_t x = 0; x < parameters.size; x++)
      {
        initialTemperature[y * parameters.size + x] = parameters.coolerTemperature;
      }
    }
  }// end of parallel

  //set temperature for heater
  for (std::size_t x = 0; x < 2 * parameters.size; x++)
  { // where is cooper, set Heater
    initialTemperature[x] = (domainMap[x] == 2) ? parameters.heaterTemperature : parameters.coolerTemperature;
  }
}// end of generateData
//------------------------------------------------------------------------------


/**
 * Store data in the file
 * @param [in] DomainMap
 * @param [in] DomainParameters
 * @param [in] InitialTemperature
 */
void storeData(const int* domainMap, const float* domainParameters, const float* initialTemperature)
{
  hid_t file = H5Fcreate(parameters.fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t scalarSize[1] {1};
  hsize_t domainSize[2] {parameters.size,parameters.size};

  long size = static_cast<long>(parameters.size);

  H5LTmake_dataset_long(file,  "/EdgeSize",           1, scalarSize, &size);
  H5LTmake_dataset_float(file, "/CoolerTemp",         1, scalarSize, &parameters.coolerTemperature);
  H5LTmake_dataset_float(file, "/HeaterTemp",         1, scalarSize, &parameters.heaterTemperature);
  H5LTmake_dataset_int(file,   "/DomainMap",          2, domainSize, domainMap);
  H5LTmake_dataset_float(file, "/DomainParameters",   2, domainSize, domainParameters);
  H5LTmake_dataset_float(file, "/InitialTemperature", 2, domainSize, initialTemperature);

  H5Fclose(file);
}// end of storeData
//------------------------------------------------------------------------------

/**
 * main function
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char** argv)
{
  parseCommandline(argc,argv);

  fmt::print("---------------------------------------------\n");
  fmt::print("--------- PPP 2020 data generator -----------\n");
  fmt::print("---------------------------------------------\n");
  fmt::print("File name:   {}\n", parameters.fileName);
  fmt::print("Size:        [{},{}]\n", parameters.size, parameters.size);
  fmt::print("Heater temp: {:.2f} °C\n", parameters.heaterTemperature);
  fmt::print("Cooler temp: {:.2f}\n", parameters.coolerTemperature);

  auto domainMap          = std::make_unique<int[]>(parameters.size * parameters.size);
  auto domainParameters   = std::make_unique<float[]>(parameters.size * parameters.size);
  auto initialTemperature = std::make_unique<float[]>(parameters.size * parameters.size);

  fmt::print("Air:         {:f}\n", air.getF0(parameters.dx,parameters.dt));
  fmt::print("Aluminum:    {:f}\n", aluminum.getF0(parameters.dx,parameters.dt));
  fmt::print("Copper:      {:f}\n", copper.getF0(parameters.dx,parameters.dt));

  if (!(copper.checkStability(parameters.dx,parameters.dt) &&
        aluminum.checkStability(parameters.dx,parameters.dt) &&
        air.checkStability(parameters.dx,parameters.dt)))
  {
    fmt::print("dt and dx are too big, simulation may be unstable!\n");
  }

  fmt::print("---------------------------------------------\n");

  fmt::print("Generating data... ");
  generateData(domainMap.get(), domainParameters.get(), initialTemperature.get());
  fmt::print("done\n");

  fmt::print("Storing data... ");
  storeData(domainMap.get(), domainParameters.get(), initialTemperature.get());
  fmt::print("done\n");

  fmt::print("---------------------------------------------\n");
}// end of main
//------------------------------------------------------------------------------
