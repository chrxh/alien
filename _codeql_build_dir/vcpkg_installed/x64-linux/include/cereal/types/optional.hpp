/*! \file optional.hpp
    \brief Support for std::optional
    \ingroup STLSupport */
/*
  Copyright (c) 2017, Juan Pedro Bolivar Puente
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the copyright holder nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef CEREAL_TYPES_STD_OPTIONAL_
#define CEREAL_TYPES_STD_OPTIONAL_

#include "cereal/cereal.hpp"
#include <optional>

namespace cereal {
  //! Saving for std::optional
  template <class Archive, typename T> inline
  void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const std::optional<T>& optional)
  {
    if(!optional) {
      ar(CEREAL_NVP_("nullopt", true));
    } else {
      ar(CEREAL_NVP_("nullopt", false),
         CEREAL_NVP_("data", *optional));
    }
  }

  //! Loading for std::optional
  template <class Archive, typename T> inline
  void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, std::optional<T>& optional)
  {
    bool nullopt;
    ar(CEREAL_NVP_("nullopt", nullopt));

    if (nullopt) {
      optional = std::nullopt;
    } else {
      optional.emplace();
      ar(CEREAL_NVP_("data", *optional));
    }
  }
} // namespace cereal

#endif // CEREAL_TYPES_STD_OPTIONAL_
