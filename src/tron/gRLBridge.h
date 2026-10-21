/*

*************************************************************************

ArmageTron -- Just another Tron Lightcycle Game in 3D.
Copyright (C) 2026  Armagetron Advanced Team

**************************************************************************

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

***************************************************************************

*/

#ifndef ArmageTron_RLBRIDGE_H
#define ArmageTron_RLBRIDGE_H

#include "defs.h"

#include <vector>

class eGrid;

//! Lockstep bridge that lets an external process (reinforcement learning trainer)
//! observe the game and steer selected AI players. Inactive unless RL_BRIDGE_SOCKET is set.
//! Protocol: see rl/PROTOCOL.md.
namespace gRLBridge
{
//! state of one objective zone, as reported by zone modules
struct ZoneInfo
{
    int team;        //!< owning team id, -1 if none
    REAL x, y;       //!< center
    REAL radius;     //!< current effective radius
    REAL conquered;  //!< conquest progress, 0 (safe) to 1 (conquered)
    bool safe;       //!< not yet conquered
};

typedef void ( *ZoneReporter )( eGrid const * grid, std::vector< ZoneInfo > & zones );

//! lets a zone implementation contribute to observations
void RegisterZoneReporter( ZoneReporter reporter );

//! called every simulation step on the server
void Timestep( eGrid * grid, REAL time );

//! true if the bridge is configured
bool Enabled();
} // namespace gRLBridge

#endif
