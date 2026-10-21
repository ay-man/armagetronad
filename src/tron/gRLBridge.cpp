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

#include "gRLBridge.h"

#include "gAIBase.h"
#include "gCycle.h"
#include "gGame.h"
#include "gSensor.h"
#include "eGrid.h"
#include "eLadderLog.h"
#include "ePlayer.h"
#include "eTeam.h"
#include "tConfiguration.h"
#include "tConsole.h"
#include "tString.h"
#include "uMenu.h"

#include <cmath>
#include <sstream>
#include <string>

#ifndef WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#endif

static tString sg_rlSocket( "" );
static tSettingItem< tString > sg_rlSocketConf( "RL_BRIDGE_SOCKET", sg_rlSocket );

// AI players on teams with an id below this are steered by the bridge; negative: all AI players
static int sg_rlTeams = 1;
static tSettingItem< int > sg_rlTeamsConf( "RL_BRIDGE_TEAMS", sg_rlTeams );

static REAL sg_rlDecisionInterval = .1;
static tSettingItem< REAL > sg_rlDecisionIntervalConf( "RL_DECISION_INTERVAL", sg_rlDecisionInterval );

static int sg_rlRays = 16;
static tSettingItem< int > sg_rlRaysConf( "RL_RAYS", sg_rlRays );

static REAL sg_rlRayRange = 250;
static tSettingItem< REAL > sg_rlRayRangeConf( "RL_RAY_RANGE", sg_rlRayRange );

static const int sg_rlProtocolVersion = 1;

namespace
{
std::vector< gRLBridge::ZoneReporter > & ZoneReporters()
{
    static std::vector< gRLBridge::ZoneReporter > reporters;
    return reporters;
}

struct BridgeState
{
    int fd = -1;
    bool failed = false;
    bool roundActive = false;
    int round = 0;
    REAL nextDecision = 0;
    std::vector< std::string > events;
    std::string readBuffer;
};

BridgeState & State()
{
    static BridgeState state;
    return state;
}

bool IsSteered( ePlayerNetID const * player )
{
    if ( !player || !dynamic_cast< gAIPlayer const * >( player ) )
        return false;
    eTeam * team = player->CurrentTeam();
    return team && ( sg_rlTeams < 0 || team->TeamID() < sg_rlTeams );
}

//! does nothing; keeps the built-in AI from driving bridge controlled cycles
class gRLPuppet : public gSimpleAI
{
protected:
    REAL DoThink() override
    {
        return 1;
    }
};

class gRLPuppetFactory : public gSimpleAIFactory
{
public:
    gRLPuppetFactory()
    {
        Set( this );
    }

protected:
    gSimpleAI * DoCreate( gCycle * object ) const override
    {
        if ( !gRLBridge::Enabled() || !object || !IsSteered( object->Player() ) )
            return NULL;
        return new gRLPuppet;
    }
};

gRLPuppetFactory sg_puppetFactory;

void OnLadderLog( std::string const & line )
{
    State().events.push_back( line );
}

void Fail( char const * what )
{
    BridgeState & state = State();
    con << "RL bridge: " << what;
#ifndef WIN32
    if ( errno )
        con << " (" << strerror( errno ) << ")";
#endif
    con << ", shutting down.\n";
    state.failed = true;
    if ( state.fd >= 0 )
    {
#ifndef WIN32
        close( state.fd );
#endif
        state.fd = -1;
    }
    uMenu::quickexit = uMenu::QuickExit_Total;
}

bool Connect()
{
    BridgeState & state = State();
#ifdef WIN32
    errno = 0;
    Fail( "not supported on this platform" );
    return false;
#else
    state.fd = socket( AF_UNIX, SOCK_STREAM, 0 );
    if ( state.fd < 0 )
    {
        Fail( "could not create socket" );
        return false;
    }

    sockaddr_un address;
    memset( &address, 0, sizeof( address ) );
    address.sun_family = AF_UNIX;
    if ( sg_rlSocket.size() >= sizeof( address.sun_path ) )
    {
        errno = 0;
        Fail( "socket path too long" );
        return false;
    }
    strncpy( address.sun_path, sg_rlSocket.c_str(), sizeof( address.sun_path ) - 1 );

    if ( connect( state.fd, reinterpret_cast< sockaddr * >( &address ), sizeof( address ) ) != 0 )
    {
        Fail( "could not connect to trainer" );
        return false;
    }

    eLadderLogWriter::SetListener( &OnLadderLog );
    con << "RL bridge: connected to " << sg_rlSocket << ".\n";
    return true;
#endif
}

bool SendLine( std::string const & line )
{
#ifndef WIN32
    BridgeState & state = State();
    std::string data = line + '\n';
    size_t sent = 0;
    while ( sent < data.size() )
    {
        ssize_t n = send( state.fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL );
        if ( n <= 0 )
        {
            if ( n < 0 && errno == EINTR )
                continue;
            Fail( "lost connection while sending" );
            return false;
        }
        sent += n;
    }
#endif
    return true;
}

bool ReadLine( std::string & line )
{
#ifndef WIN32
    BridgeState & state = State();
    for ( ;; )
    {
        size_t end = state.readBuffer.find( '\n' );
        if ( end != std::string::npos )
        {
            line = state.readBuffer.substr( 0, end );
            state.readBuffer.erase( 0, end + 1 );
            return true;
        }

        char buffer[4096];
        ssize_t n = recv( state.fd, buffer, sizeof( buffer ), 0 );
        if ( n <= 0 )
        {
            if ( n < 0 && errno == EINTR )
                continue;
            if ( n == 0 )
                errno = 0;
            Fail( "lost connection while waiting for the trainer" );
            return false;
        }
        state.readBuffer.append( buffer, n );
    }
#else
    return false;
#endif
}

std::string JsonString( std::string const & s )
{
    std::ostringstream o;
    o << '"';
    for ( unsigned char c : s )
    {
        if ( c == '"' || c == '\\' )
            o << '\\' << c;
        else if ( c < 0x20 )
            o << ' ';
        else
            o << c;
    }
    o << '"';
    return o.str();
}

// finite numbers only, JSON has no NaN
REAL Finite( REAL x )
{
    return std::isfinite( x ) ? x : 0;
}

eCoord Normalized( eCoord const & v )
{
    REAL len = sqrt( v.NormSquared() );
    return len > 0 ? v * ( 1 / len ) : eCoord( 1, 0 );
}

void WriteCycle( std::ostream & o, ePlayerNetID * player, gCycle * cycle, bool withSenses )
{
    eTeam * team = player->CurrentTeam();
    o << "{\"id\":" << player->ID()
      << ",\"team\":" << ( team ? team->TeamID() : -1 )
      << ",\"name\":" << JsonString( player->GetUserName() )
      << ",\"alive\":" << ( cycle && cycle->Alive() ? 1 : 0 );

    if ( cycle )
    {
        eCoord pos = cycle->Position();
        eCoord dir = Normalized( cycle->Direction() );
        o << ",\"x\":" << Finite( pos.x ) << ",\"y\":" << Finite( pos.y )
          << ",\"dx\":" << Finite( dir.x ) << ",\"dy\":" << Finite( dir.y )
          << ",\"speed\":" << Finite( cycle->Speed() )
          << ",\"braking\":" << ( cycle->GetBraking() ? 1 : 0 );

        if ( withSenses )
        {
            o << ",\"rubber\":" << Finite( cycle->GetRubber() / ( sg_rubberCycle > 0 ? sg_rubberCycle : 1 ) )
              << ",\"brake_reservoir\":" << Finite( cycle->GetBrakingReservoir() )
              << ",\"can_turn_left\":" << ( cycle->CanMakeTurn( -1 ) ? 1 : 0 )
              << ",\"can_turn_right\":" << ( cycle->CanMakeTurn( 1 ) ? 1 : 0 )
              << ",\"rays\":[";

            int rays = sg_rlRays > 0 ? sg_rlRays : 1;
            for ( int i = 0; i < rays; ++i )
            {
                REAL angle = 2 * M_PI * i / rays;
                eCoord rayDir( dir.x * cos( angle ) - dir.y * sin( angle ), dir.x * sin( angle ) + dir.y * cos( angle ) );
                REAL dist = sg_rlRayRange;
                int type = gSENSOR_NONE;
                if ( cycle->Alive() )
                {
                    gSensor sensor( cycle, pos, rayDir );
                    sensor.detect( sg_rlRayRange );
                    if ( sensor.ehit )
                    {
                        dist = sensor.hit < sg_rlRayRange ? sensor.hit : sg_rlRayRange;
                        type = sensor.type;
                    }
                }
                o << ( i ? "," : "" ) << '[' << Finite( dist ) << ',' << type << ']';
            }
            o << ']';
        }
    }
    o << '}';
}

std::string BuildMessage( eGrid * grid, char const * type, REAL time )
{
    BridgeState & state = State();
    std::ostringstream o;
    o.precision( 6 );
    o << "{\"type\":\"" << type << "\",\"version\":" << sg_rlProtocolVersion
      << ",\"round\":" << state.round << ",\"time\":" << Finite( time )
      << ",\"winner\":" << sg_RoundWinner()
      << ",\"decision_interval\":" << sg_rlDecisionInterval
      << ",\"ray_range\":" << sg_rlRayRange;

    // steered agents carry their senses, everybody else just their kinematics
    o << ",\"agents\":[";
    bool first = true;
    for ( int i = 0; i < se_PlayerNetIDs.Len(); ++i )
    {
        ePlayerNetID * player = se_PlayerNetIDs( i );
        if ( !IsSteered( player ) )
            continue;
        o << ( first ? "" : "," );
        first = false;
        WriteCycle( o, player, dynamic_cast< gCycle * >( player->Object() ), true );
    }
    o << "],\"cycles\":[";
    first = true;
    for ( int i = 0; i < se_PlayerNetIDs.Len(); ++i )
    {
        ePlayerNetID * player = se_PlayerNetIDs( i );
        if ( !player->CurrentTeam() )
            continue;
        o << ( first ? "" : "," );
        first = false;
        WriteCycle( o, player, dynamic_cast< gCycle * >( player->Object() ), false );
    }

    o << "],\"teams\":[";
    for ( int i = 0; i < eTeam::teams.Len(); ++i )
    {
        eTeam * team = eTeam::teams( i );
        o << ( i ? "," : "" ) << "{\"id\":" << team->TeamID() << ",\"score\":" << team->Score()
          << ",\"alive\":" << ( team->Alive() ? 1 : 0 ) << ",\"name\":" << JsonString( team->GetLogName() ) << '}';
    }

    std::vector< gRLBridge::ZoneInfo > zones;
    for ( auto reporter : ZoneReporters() )
        reporter( grid, zones );
    o << "],\"zones\":[";
    for ( size_t i = 0; i < zones.size(); ++i )
    {
        gRLBridge::ZoneInfo const & z = zones[i];
        o << ( i ? "," : "" ) << "{\"team\":" << z.team << ",\"x\":" << Finite( z.x ) << ",\"y\":" << Finite( z.y )
          << ",\"radius\":" << Finite( z.radius ) << ",\"conquered\":" << Finite( z.conquered )
          << ",\"safe\":" << ( z.safe ? 1 : 0 ) << '}';
    }

    o << "],\"events\":[";
    for ( size_t i = 0; i < state.events.size(); ++i )
        o << ( i ? "," : "" ) << JsonString( state.events[i] );
    state.events.clear();
    o << "]}";
    return o.str();
}

//! reply format: "A <id> <turn -1|0|1> <brake 0|1> ..." or "OK" or "QUIT"
bool ApplyReply( std::string const & line )
{
    std::istringstream s( line );
    std::string command;
    s >> command;
    if ( command == "QUIT" )
    {
        con << "RL bridge: trainer requested shutdown.\n";
        uMenu::quickexit = uMenu::QuickExit_Total;
        return true;
    }
    if ( command == "OK" )
        return true;
    if ( command != "A" )
    {
        errno = 0;
        Fail( ( "malformed reply '" + line.substr( 0, 80 ) + "'" ).c_str() );
        return false;
    }

    int id, turn, brake;
    while ( s >> id >> turn >> brake )
    {
        ePlayerNetID * target = NULL;
        for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
        {
            if ( se_PlayerNetIDs( i )->ID() == id )
                target = se_PlayerNetIDs( i );
        }
        if ( !IsSteered( target ) )
        {
            con << "RL bridge: ignoring action for player " << id << ", it is not steered.\n";
            continue;
        }
        gCycle * cycle = dynamic_cast< gCycle * >( target->Object() );
        if ( !cycle || !cycle->Alive() )
            continue;

        if ( turn < 0 )
            cycle->Act( &gCycle::se_turnLeft, 1 );
        else if ( turn > 0 )
            cycle->Act( &gCycle::se_turnRight, 1 );
        cycle->Act( &gCycle::s_brake, brake ? 1 : 0 );
    }
    return true;
}

bool Exchange( eGrid * grid, char const * type, REAL time )
{
    if ( !SendLine( BuildMessage( grid, type, time ) ) )
        return false;
    std::string reply;
    if ( !ReadLine( reply ) )
        return false;
    return ApplyReply( reply );
}

bool AnySteeredAlive()
{
    for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
    {
        ePlayerNetID * player = se_PlayerNetIDs( i );
        if ( IsSteered( player ) && player->Object() && player->Object()->Alive() )
            return true;
    }
    return false;
}
} // namespace

void gRLBridge::RegisterZoneReporter( ZoneReporter reporter )
{
    ZoneReporters().push_back( reporter );
}

bool gRLBridge::Enabled()
{
    return sg_rlSocket.size() > 0;
}

void gRLBridge::Timestep( eGrid * grid, REAL time )
{
    BridgeState & state = State();
    if ( !Enabled() || state.failed )
        return;
    if ( state.fd < 0 && !Connect() )
        return;

    // nothing moves during the countdown
    if ( time < 0 )
        return;

    int winner = sg_RoundWinner();
    if ( state.roundActive )
    {
        // a new round started without the old one getting decided
        bool restarted = false;
        for ( auto const & event : state.events )
            restarted = restarted || event.compare( 0, 9, "NEW_ROUND" ) == 0;

        if ( winner != 0 || restarted )
        {
            state.roundActive = false;
            Exchange( grid, "round_end", time );
            return;
        }
    }
    else
    {
        if ( winner != 0 || !AnySteeredAlive() )
            return;
        state.roundActive = true;
        state.round++;
        state.nextDecision = time;
    }

    if ( time >= state.nextDecision )
    {
        state.nextDecision += sg_rlDecisionInterval;
        if ( state.nextDecision < time )
            state.nextDecision = time + sg_rlDecisionInterval;
        Exchange( grid, "step", time );
    }
}
