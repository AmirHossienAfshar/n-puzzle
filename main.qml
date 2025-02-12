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
    property var previousPuzzleList: bridge.pyside_puzzle_list.slice(0)  // Stores old positions

    GridView {
        id: puzzleGrid
        anchors.centerIn: parent
        width: parent.width
        height: parent.height
        cellWidth: tileSize
        cellHeight: tileSize
        model: bridge.pyside_puzzle_list
        interactive: false  // The user cannot interact, AI controls movement

        delegate: Rectangle {
            id: tile
            width: puzzleGrid.cellWidth
            height: puzzleGrid.cellHeight
            color: modelData === 0 ? "black" : "white"
            border.color: "black"

            Text {
                anchors.centerIn: parent
                text: modelData === 0 ? "" : modelData
                font.pixelSize: 24
            }

            // Behavior on x { NumberAnimation { duration: 200 } }
            // // Behavior on y { NumberAnimation { duration: 200 } }

            // Component.onCompleted: updatePosition()

            // Connections {
            //     target: bridge
            //     function onPyside_puzzle_listChanged() {
            //         moveTile()
            //     }
            // }

            // function updatePosition() {
            //     let index = model.index
            //     let row = Math.floor(index / gridSize)
            //     let col = index % gridSize
            //     x = col * puzzleGrid.cellWidth
            //     y = row * puzzleGrid.cellHeight
            // }

            // function moveTile() {
            //     if (previousPuzzleList.length !== bridge.pyside_puzzle_list.length) {
            //         previousPuzzleList = bridge.pyside_puzzle_list.slice(0);
            //         return;
            //     }

            //     let newIndex = bridge.pyside_puzzle_list.indexOf(modelData)
            //     if (newIndex !== -1) {
            //         let oldIndex = previousPuzzleList.indexOf(modelData)
            //         if (oldIndex !== newIndex) {  // If tile has moved
            //             let newRow = Math.floor(newIndex / gridSize)
            //             let newCol = newIndex % gridSize
            //             x = newCol * puzzleGrid.cellWidth
            //             y = newRow * puzzleGrid.cellHeight
            //         }
            //     }

            //     previousPuzzleList = bridge.pyside_puzzle_list.slice(0); // Store state for next move
            // }
        }
    }
}
