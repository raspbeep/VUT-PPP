/**
 * @file    SequentialHeatSolver.hpp
 * 
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *          David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of sequential heat equation solver.
 *
 * @date    2024-02-23
 */

#ifndef SEQUENTIAL_HEAT_SOLVER_HPP
#define SEQUENTIAL_HEAT_SOLVER_HPP

#include <vector>

#include <hdf5.h>

#include "AlignedAllocator.hpp"
#include "Hdf5Handle.hpp"
#include "HeatSolverBase.hpp"

/**
 * @brief The SequentialHeatSolver class implements reference sequential heat
 *        equation solver in 2D domain.
 */
class SequentialHeatSolver : public HeatSolverBase
{
  public:
    /**
     * @brief Constructor - Initializes the solver. This includes:
     *        - Allocate temporary working storage for simulation.
     *        - Open output HDF5 file (if filename was provided).
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    SequentialHeatSolver(const SimulationProperties& simulationProps, const MaterialProperties& materialProps);
    
    /// @brief inherit constructor from base class
    using HeatSolverBase::HeatSolverBase;

    /// @brief Destructor
    ~SequentialHeatSolver() override = default;

    /// @brief Inherit assignment operator from base class
    using HeatSolverBase::operator=;

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     */
    void run(std::vector<float, AlignedAllocator<float>>& outResult) override;

  protected:
  private:
    /**
     * @brief Get type of the code.
     * @return Returns type of the code.
     */
    std::string_view getCodeType() const override;
    
    /**
     * @brief Compute average temperature in middle column of the domain.
     * @param data 2D array containing simulation state.
     * @return Returns average temperature in the middle column of the domain.
     */
    float computeMiddleColAvgTemp(const float* data) const;

    static constexpr std::string_view codeType{"seq"};  ///< Type of the code.

    std::vector<float, AlignedAllocator<float>> mTempArray;   ///< Temporary work array.
    Hdf5FileHandle                              mFileHandle;  ///< Output HDF5 file handle.
};

#endif /* SEQUENTIAL_HEAT_SOLVER_HPP */
