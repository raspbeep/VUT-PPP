/**
 * @file    MaterialProperties.hpp
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

#ifndef MATERIAL_PROPERTIES_HPP
#define MATERIAL_PROPERTIES_HPP

#include <cstddef>
#include <string_view>
#include <vector>

#include "AlignedAllocator.hpp"

/**
 * @brief The MaterialProperties class represents the simulation domain and its
 *        contents.
 */
class MaterialProperties
{
  public:
    /// @brief Default constructor
    MaterialProperties() = default;

    /**
     * @brief Copy constructor
     * @param other
     */
    explicit MaterialProperties(const MaterialProperties& other) = default;

    /**
     * @brief Move constructor
     * @param other
     */
    explicit MaterialProperties(MaterialProperties&& other) = default;
    
    /// @brief Destructor
    ~MaterialProperties() = default;

    /**
     * @brief Copy assignment operator
     * @param other
     * @return reference to this
     */
    MaterialProperties &operator=(const MaterialProperties& other) = default;

    /**
     * @brief Move assignment operator
     * @param other
     * @return reference to this
     */
    MaterialProperties &operator=(MaterialProperties&& other) = default;

    /**
     * @brief load domain information from the input material file.
     * @param fileName Path to material file.
     * @param loadData Flag which specifies if we want actually load contents
     *                 of the domain (materials) or just meta-data.
     *                 NOTE: When "false" is specified, vectors: DomainMap,
     *                       DomainParams and InitTemp stay empty!
     */
    void load(std::string_view fileName, bool loadData);

    /**
     * @brief Getter for domain map.
     * @return Domain map.
     */
    const std::vector<int, AlignedAllocator<int>>& getDomainMap() const;

    /**
     * @brief Getter for domain parameters.
     * @return Domain parameters.
     */
    const std::vector<float, AlignedAllocator<float>>& getDomainParameters() const;

    /**
     * @brief Getter for initial temperature distribution.
     * @return Initial temperature distribution.
     */
    const std::vector<float, AlignedAllocator<float>>& getInitialTemperature() const;

    /**
     * @brief Getter for the temperature of the cooler.
     * @return Temperature of the cooler.
     */
    float getCoolerTemperature() const;

    /**
     * @brief Getter for the temperature of the heater.
     * @return Temperature of the heater.
     */
    float getHeaterTemperature() const;

    /**
     * @brief Getter for the size of the edge of the domain.
     * @return Size of the edge of the domain.
     */
    std::size_t getEdgeSize() const;

    /**
     * @brief Getter for the total number of gridpoints in the domain.
     * @return Total number of gridpoints in the domain.
     */
    std::size_t getGridPointCount() const;

  protected:
  private:
    /**
     * @brief Domain Map - defines type of the material at every gridpoint
     *        0 - air, 1 - aluminium, 2 - copper.
     */
    std::vector<int, AlignedAllocator<int>>     mDomainMap{};
    std::vector<float, AlignedAllocator<float>> mDomainParams{};   ///< Thermal properties of the medium.
    std::vector<float, AlignedAllocator<float>> mInitTemp{};       ///< Initial temperature distribution.

    float                                       mCoolerTemp{};     ///< Temperature of the air.
    float                                       mHeaterTemp{};     ///< Temperature of the heater.
    std::size_t                                 mEdgeSize{};       ///< Size of the domain.
    std::size_t                                 mGridPointCount{}; ///< Total number of gridpoint in the domain.
};

#endif /* MATERIAL_PROPERTIES_HPP */
