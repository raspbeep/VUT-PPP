/**
 * @file    HeatSolverBase.hpp
 * 
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *          David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains base class from which sequential and parallel
 *          versions of the solver are derived.
 *
 * @date    2024-02-22
 */

#ifndef HEAT_SOLVER_BASE_HPP
#define HEAT_SOLVER_BASE_HPP

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <hdf5.h>

#include "AlignedAllocator.hpp"
#include "MaterialProperties.hpp"
#include "SimulationProperties.hpp"

/**
 * @brief The HeatSolverBase class represents base class for implementation of
 * simple heat equation solver in heterogeneous medium.
 */
class HeatSolverBase
{
  public:
    /// @brief Default constructor is deleted
    HeatSolverBase() = delete;
    
    /// @brief Copy constructor is deleted
    HeatSolverBase(const HeatSolverBase&) = delete;

    /// @brief Move constructor is deleted
    HeatSolverBase(HeatSolverBase&&) = delete;

    /**
     * @brief Destructor
     */
    virtual ~HeatSolverBase() = default;

    /// @brief Copy assignment operator is deleted
    HeatSolverBase& operator=(const HeatSolverBase&) = delete;

    /// @brief Move assignment operator is deleted
    HeatSolverBase& operator=(HeatSolverBase&&) = delete;

    /**
     * @brief Pure-virtual method that needs to be implemented by each solver
     * implementation and which is called to execute the simulation.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: Note that the vector can be 0-sized in case of the
     *                  MPI based implementation (where only ROOT - rank = 0 -
     *                  process gets needs this output array)!
     */
    virtual void run(std::vector<float, AlignedAllocator<float>>& outResult) = 0;

  protected:
    /**
     * @brief Evalute heat-equation solver stencil function at the specified point
     *        using 4-neighbourhood and 2 points in each direction. The method
     *        uses results of previous simulation step and writtes new value into
     *        the next one.
     *
     * @param oldTemp     [IN]  Array representing the domain state computed in PREVIOUS sim. step.
     * @param newTemp     [OUT] Array representing the domain in which computed point will be stored.
     * @param params      Parameters of the material at each point of the (sub-)domain.
     * @param map         Material map where "0" values represent air.
     * @param i           Row (Y-axis) position of the evaluated point (as in i-th row).
     * @param j           Column (X-axis) position of the evaluated point (as in j-th column).
     * @param width       Width (or row length) of "oldTemp" and "newTemp" 2D arrays.
     * @param airFlowRate Rate of heat dissipation due to air flow.
     * @param coolerTemp  Temperature of the cooler.
     */
    static constexpr void computePoint(const float* oldTemp,
                                       float*       newTemp,
                                       const float* params,
                                       const int*   map,
                                       std::size_t  i,
                                       std::size_t  j,
                                       std::size_t  width,
                                       float        airflowRate,
                                       float        coolerTemp);

    /**
     * @brief Constructor
     * @param simulationProps Parameters of simulation read from command line arguments.
     * @param materialProps   Parameters of material read from the input file.
     */
    HeatSolverBase(const SimulationProperties& simulationProps, const MaterialProperties& materialProps);

    /**
     * @brief Returns type of the executed code (sequential or parallel).
     * @return Returns "seq" for sequential code and "par" for parallel (MPI or Hybrid) code.
     */
    virtual std::string_view getCodeType() const = 0;

    /**
     * @brief Returns "true" every N-th iteration so that "true" is returned every
     *        10% of the progress.
     * @param iteration Integer representing current iteration.
     * @return Returns "true" when the code is supposed to print progress report.
     */
    bool shouldPrintProgress(std::size_t iteration) const;

    /**
     * @brief Returns "true" if the current iteration data should be stored.
     * @param iteration Integer representing current iteration.
     * @return Returns "true" if the current iteration data should be stored.
     */
    bool shouldStoreData(std::size_t iteration) const;

    /**
     * @brief Prints human readable simulation progress report every N-th iteration
     *        (see "shouldPrintProgress" which is used internally).
     * @param iteration        Integer representing current iteration.
     * @param middleColAvgTemp Computed temperature average of the column
     *                         in the middle of the domain.
     */
    void printProgressReport(std::size_t iteration, float middleColAvgTemp);

    /**
     * @brief Print either human or machine (CSV) readable report of status of
     *        FINISHED simulation.
     *        NOTE: Should be called after last iteration of the simulation.
     * @param totalTime        Total time elapsed since the simulation begun [s] (ie. MPI_Wtime(...))
     * @param middleColAvgTemp Computed temperature average of the middle column in the domain.
     */
    void printFinalReport(double totalTime, float middleColAvgTemp) const;

    /**
     * @brief Stores the simulation state in "data" into HDF5 file using sequential HDF5
     *        this *HAS* to be called only from single (usually MASTER/RANK=0 process).
     *        The method assumes that the "data" is 2D square array of "edgeSize"x"edgeSize"
     *        elements (where "edgeSize" is read from the input file).
     * @param fileHandle Handle to opened HDF5 file (using sequential HDF5).
     * @param iteration  Integer representing current iteration.
     * @param data       2D square array of "edgeSize"x"edgeSize" elements.
     */
    void storeDataIntoFile(hid_t fileHandle, std::size_t iteration, const float* data);

    /**
     * @brief Evaluate heat-equation over specified tile
     * @param oldTemp       [IN]  Array representing the domain state computed in PREVIOUS sim. step.
     * @param newTemp       [OUT] Array representing the updated domain.
     * @param params        Parameters of the material at each point of the (sub-)domain.
     * @param map           Material map where "0" values represent air.
     * @param offsetX       Offset of updated region of the array in X direction (>= 2).
     * @param offsetY       Offset of updated region of the array in Y direction (>= 2).
     * @param sizeX         Size of the updated region of the array in X direction.
     * @param sizeY         Size of the updated region of the array in Y direction.
     * @param strideX       Total size of the array in X direction.
     * @param airFlowRate   Rate of heat dissipation due to air flow.
     * @param coolerTemp    Temperature of the cooler.
     */
    void updateTile(const float* oldTemp,
                    float*       newTemp,
                    const float* params,
                    const int*   map,
                    std::size_t  offsetX,
                    std::size_t  offsetY,
                    std::size_t  sizeX,
                    std::size_t  sizeY,
                    std::size_t  strideX) const;

    const SimulationProperties& mSimulationProps; ///< Parameters of the simulation (from arguments)
    const MaterialProperties&   mMaterialProps;   ///< Parameters of the material (from input file)
  private:
};

#pragma omp declare simd notinbranch \
                         uniform(oldTemp, newTemp, params, map, width, airflowRate, coolerTemp) \
                         linear(i, j)
inline constexpr void HeatSolverBase::computePoint(const float* oldTemp,
                                                   float*       newTemp,
                                                   const float* params,
                                                   const int*   map,
                                                   std::size_t  i,
                                                   std::size_t  j,
                                                   std::size_t  width,
                                                   float        airflowRate,
                                                   float        coolerTemp)
{
  // 1. Precompute neighbor indices.
  const unsigned center    = static_cast<unsigned>(i * width + j);
  const unsigned top[2]    = { center - static_cast<unsigned>(width), center - 2 * static_cast<unsigned>(width) };
  const unsigned bottom[2] = { center + static_cast<unsigned>(width), center + 2 * static_cast<unsigned>(width) };
  const unsigned left[2]   = { center - 1, center - 2 };
  const unsigned right[2]  = { center + 1, center + 2 };

  // 2. The reciprocal value of the sum of domain parameters for normalization.
  const float frac = 1.0f / (params[top[0]]    + params[top[1]]    +
                             params[bottom[0]] + params[bottom[1]] +
                             params[left[0]]   + params[left[1]]   +
                             params[right[0]]  + params[right[1]]  +
                             params[center]);

  // 3. Compute new temperature at the specified grid point.
  float pointTemp = oldTemp[top[0]]    * params[top[0]]    * frac +
                    oldTemp[top[1]]    * params[top[1]]    * frac +
                    oldTemp[bottom[0]] * params[bottom[0]] * frac +
                    oldTemp[bottom[1]] * params[bottom[1]] * frac +
                    oldTemp[left[0]]   * params[left[0]]   * frac +
                    oldTemp[left[1]]   * params[left[1]]   * frac +
                    oldTemp[right[0]]  * params[right[0]]  * frac +
                    oldTemp[right[1]]  * params[right[1]]  * frac +
                    oldTemp[center]    * params[center]    * frac;

  // 4. Remove some of the heat due to air flow
  pointTemp = (map[center] == 0)
                ? (airflowRate * coolerTemp) + ((1.0f - airflowRate) * pointTemp)
                : pointTemp;

  newTemp[center] = pointTemp;
}

#endif /* HEAT_SOLVER_BASE_HPP */
