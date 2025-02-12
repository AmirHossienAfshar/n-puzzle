import QtQuick 2.15
import QtQuick.Controls 2.15
import Pyside_Bridge 1.0

ApplicationWindow {
    visible: true
    width: 400
    height: 400
    title: "Sliding Puzzle (AI-Controlled)"

    Pyside_Bridge_class {
        id: bridge
    }

    property int gridSize: bridge.pyside_puzzle_size
    property real tileSize: width / gridSize

    PuzzleBoard {
        // width: 400
        // height: 400
        id: board
        model: bridge.pyside_puzzle_list
        tileSize: width / bridge.pyside_puzzle_size
    }
}