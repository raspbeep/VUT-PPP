/**
 * @file    SimulationProperties.hpp
 * 
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *          David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          Parameters of the simulation passed as application arguments.
 *
 * @date    2024-02-22
 */

#ifndef SIMULATION_PROPERTIES_HPP
#define SIMULATION_PROPERTIES_HPP

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include "MaterialProperties.hpp"

/**
 * @brief The SimulationProperties class represents parameters of the simulation
 *        passed as application arguments.
 */
class SimulationProperties
{
  public:
    /**
     * @brief Execution of the simulation
     */
    enum class Execution
    {
      seq     = 0, /// @brief Run sequential code only.
      par_p2p = 1, /// @brief Run parallel solver using poit-to-point.
      par_rma = 2, /// @brief Run parallel solver using RMA.
    };

    /**
     * @brief Type of the decomposition in MPI code
     */
    enum class Decomposition
    {
      d1 = 0, /// @brief Run in 1D decomposition (= Nx1).
      d2 = 1, /// @brief Run in 2D decomposition (~ N^(1/2) x N^(1/2)).
    };

    /**
     * @brief Constructor
     */
    SimulationProperties();
    
    /**
     * @brief Copy constructor
     * @param other Object to be copied.
     */
    explicit SimulationProperties(const SimulationProperties& other) = default;

    /**
     * @brief Move constructor
     * @param other Object to be moved.
     */
    SimulationProperties(SimulationProperties&& other) = default;

    /**
     * @brief Destructor
     */
    ~SimulationProperties() = default;

    /**
     * @brief Copy assignment operator
     * @param other Object to be copied.
     * @return Returns reference to the current object.
     */
    SimulationProperties& operator=(const SimulationProperties& other) = default;

    /**
     * @brief Move assignment operator
     * @param other Object to be moved.
     * @return Returns reference to the current object.
     */
    SimulationProperties& operator=(SimulationProperties&& other) = default;

    /**
     * @brief Parse command line arguments passed to the application.
     * @param argc
     * @param argv
     */
    void parseCommandLine(int argc, char* argv[]);

    /**
     * @brief Print current simulation parameters.
     * @param materialProps Properties of the domain loaded from the input file.
     */
    void printParameters(const MaterialProperties& materialProps) const;

    /**
     * @brief Run sequential version?
     * @return Returns "true" if sequential version should be executed.
     */
    bool isRunSequential() const;

    /**
     * @brief Run parallel version?
     * @return Returns "true" if parallel version should be executed.
     */
    bool isRunParallel() const;

    /**
     * @brief Run parallel P2P version?
     * @return Returns "true" if parallel version should be executed using P2P comm.
     */
    bool isRunParallelP2P() const;

    /**
     * @brief Run parallel RMA version?
     * @return Returns "true" if parallel version should be executed using RMA.
     */
    bool isRunParallelRMA() const;

    /**
     * @brief Should be validation performed?
     * @return Returns "true" if validation should be performed.
     */
    bool isValidation() const;

    /**
     * @brief Get number of iterations requested.
     * @return Returns number of iterations specified by the user.
     */
    std::size_t getNumIterations() const;

    /**
     * @brief Get number of threads per MPI process requested.
     * @return Returns number of threads per MPI process specified by the user.
     */
    int getThreadCount() const;

    /**
     * @brief Get number of iterations between each time simulation state snapshot is taken.
     * @return Returns number of iterations to skip between state snapshost.
     */
    std::size_t getWriteIntensity() const;

    /**
     * @brief Get airflow rate around the heatsink
     * @return Returns the airflow rate.
     */
    float getAirflowRate() const;

    /**
     * @brief Get path to the input material file.
     * @return Returns path to input material file.
     */
    std::string_view getMaterialFileName() const;

    /**
     * @brief Get filename and path of the output file.
     *        NOTE: User should specify code type which is creating the file as:
     *              "seq" - for sequential code
     *              "par" - for parallel code
     * @param codeTypeExt "seq" or "par" depending on which code is creating the file.
     * @return Returns path to the output file including code specific extension.
     */
    std::string getOutputFileName(std::string_view codeTypeExt = {}) const;

    /**
     * @brief Get base name for debug images.
     * @return Returns base name (if specified) of debug images.
     */
    std::string_view getDebugImageFileName() const;

    /**
     * @brief Is application running in batch mode?
     * @return Returns "true" if application should output only in batch mode.
     */
    bool isBatchMode() const;

    /**
     * @brief Is parallel I/O enabled?
     * @return Returns "true" if parallel I/O is enabled.
     */
    bool useParallelIO() const;

    /**
     * @brief isDebug
     * @return Returns "true" if debug flag was specified.
     */
    bool isDebug() const;

    /**
     * @brief GetDecompType
     * @return Returns member of "Decomposition".
     */
    Decomposition getDecomposition() const;

    /**
     * @brief Get decomposition grid
     * @return Returns number of subdivisions (tiles) in X and Y dimensions.
     */
    void getDecompGrid(int& outSizeX, int& outSizeY) const;

    /**
     * @brief Append extension to the filename before its file type extension.
     *        As in: "my_file.abc" to "my_file_EXT.abc"
     * @param fileName      Base filename (with or without extension).
     * @param fileNameExt   Extension "EXT" to be appended together with "_".
     * @param fileTypeExt   File type extension.
     * @return Returns extended filename.
     */
    static std::string appendFileNameExt(std::string_view fileName,
                                         std::string_view fileNameExt,
                                         std::string_view fileTypeExt);

  protected:
  private:
    std::size_t           mNIterations;        ///< Number of iteration of the simulation.
    int                   mThreadCount;        ///< Number of OMP threads per process.
    std::size_t           mWriteIntensity;     ///< Every N-th iteration result is stored.
    float                 mAirflowRate;        ///< Air flow rate of cooling air.

    std::string           mMaterialFileName;   ///< Path to input file.
    std::string           mOutputFileName;     ///< Path to output file.

    Execution             mExecution;          ///< Execution of the simulation.
    bool                  mDebugFlag;          ///< Compare results of sequential and parallel codes.
    bool                  mVerificationFlag;   ///< Verify the result.
    [[maybe_unused]] bool mSequentialFlag;     ///< Unused.
    bool                  mBatchMode;          ///< Output only in CSV format.
    bool                  mBatchModeHeader;    ///< Output CSV header.
    bool                  mUseParallelIO;      ///< Whether to use parallel HDF5 I/O.
    Decomposition         mDecomposition;      ///< Decomposition of the simulation in parallel mode.
    std::array<int, 2>    mGridSize;           ///< Number of tiles in each grid dimensions (N_x, N_y).

    std::string           mDebugImageName;     ///< Base filename of debug images.
};

#endif /* SIMULATION_PROPERTIES_HPP */
