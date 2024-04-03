/**
 * @file    Hdf5Handle.hpp
 * @authors David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *
 * @date    2024-02-22
 */

#ifndef HDF5_HANDLE_HPP
#define HDF5_HANDLE_HPP

#include <memory>
#include <type_traits>
#include <utility>

#include <hdf5.h>

// Forward declaration
template<auto closeFunction> class Hdf5Handle;

using Hdf5AttributeHandle    = Hdf5Handle<H5Aclose>; /// @brief A handle for a HDF5 attribute
using Hdf5DatasetHandle      = Hdf5Handle<H5Dclose>; /// @brief A handle for a HDF5 dataset
using Hdf5DataspaceHandle    = Hdf5Handle<H5Sclose>; /// @brief A handle for a HDF5 dataspace
using Hdf5FileHandle         = Hdf5Handle<H5Fclose>; /// @brief A handle for a HDF5 file
using Hdf5GroupHandle        = Hdf5Handle<H5Gclose>; /// @brief A handle for a HDF5 group
using Hdf5PropertyListHandle = Hdf5Handle<H5Pclose>; /// @brief A handle for a HDF5 property list

/**
 * @brief A class to manage HDF5 handles
 * @tparam closeFunction The function to close the handle
 */
template<auto closeFunction>
class Hdf5Handle
{
  // check if closeFunction is callable with hid_t
  static_assert(std::is_invocable_v<decltype(closeFunction), hid_t>, "closeFunction must be callable with hid_t");

  public:
    /// @brief Default constructor
    constexpr Hdf5Handle() = default;

    /**
     * @brief Constructor
     * @param handle The handle to manage
     */
    constexpr Hdf5Handle(hid_t&& handle);

    /// @brief Copy constructor not allowed
    Hdf5Handle(const Hdf5Handle&) = delete;

    /**
     * @brief Move constructor
     * @param other The other handle to move from
     */
    constexpr Hdf5Handle(Hdf5Handle&& other);

    /// @brief Destructor
    ~Hdf5Handle();

    /// @brief Copy assignment not allowed
    Hdf5Handle& operator=(const Hdf5Handle&) = delete;

    /**
     * @brief Move assignment
     * @param other The other handle to move from
     * @return A reference to this
     */
    constexpr Hdf5Handle& operator=(Hdf5Handle&& other);

    /**
     * @brief Reset the handle
     * @param handle The new handle to manage
     */
    constexpr void reset(hid_t&& handle = H5I_INVALID_HID);

    /**
     * @brief Release the handle
     * @return The handle
     */
    [[nodiscard]] constexpr hid_t release();

    /**
     * @brief Check if the handle is valid
     * @return True if the handle is valid, false otherwise
     */
    [[nodiscard]] constexpr bool valid() const;

    /**
     * @brief Conversion operator
     * @return The handle
     */
    [[nodiscard]] constexpr operator hid_t() const;
  protected:
  private:
    hid_t mHandle{H5I_INVALID_HID};
};

template<auto closeFunction>
constexpr Hdf5Handle<closeFunction>::Hdf5Handle(hid_t&& handle)
: mHandle(std::move(handle))
{}

template<auto closeFunction>
constexpr Hdf5Handle<closeFunction>::Hdf5Handle(Hdf5Handle&& other)
: mHandle(std::exchange(other.mHandle, H5I_INVALID_HID))
{}

template<auto closeFunction>
constexpr Hdf5Handle<closeFunction>& Hdf5Handle<closeFunction>::operator=(Hdf5Handle&& other)
{
  if(this != std::addressof(other))
  {
    mHandle = std::exchange(other.mHandle, H5I_INVALID_HID);
  }

  return *this;
}

template<auto closeFunction>
Hdf5Handle<closeFunction>::~Hdf5Handle()
{
  if(mHandle != H5I_INVALID_HID)
  {
    closeFunction(mHandle);
  }
}

template<auto closeFunction>
constexpr void Hdf5Handle<closeFunction>::reset(hid_t&& handle)
{
  if(mHandle != H5I_INVALID_HID)
  {
    closeFunction(mHandle);
  }

  mHandle = handle;
}

template<auto closeFunction>
constexpr hid_t Hdf5Handle<closeFunction>::release()
{
  return std::exchange(mHandle, H5I_INVALID_HID);
}

template<auto closeFunction>
constexpr bool Hdf5Handle<closeFunction>::valid() const
{
  return mHandle != H5I_INVALID_HID;
}

template<auto closeFunction>
constexpr Hdf5Handle<closeFunction>::operator hid_t() const
{
  return mHandle;
}

#endif /* HDF5_HANDLE_HPP */
