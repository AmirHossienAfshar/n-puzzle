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

    Grid {
        id: puzzleGrid
        anchors.centerIn: parent
        columns: gridSize
        rows: gridSize
        width: parent.width
        height: parent.height

        Repeater {
            model: bridge.pyside_puzzle_list.length

            delegate: Rectangle {
                id: tile
                width: tileSize
                height: tileSize
                color: modelData === 0 ? "lightgray" : "white"
                border.color: "black"

                Text {
                    anchors.centerIn: parent
                    text: modelData === 0 ? "" : modelData
                    font.pixelSize: 24
                }

                Behavior on x { NumberAnimation { duration: 150 } }
                Behavior on y { NumberAnimation { duration: 150 } }

                Component.onCompleted: updatePosition()
                Connections {
                    target: bridge
                    function onPyside_puzzle_listChanged() {
                        updatePosition()
                    }
                }

                function updatePosition() {
                    let index = model.index
                    let row = Math.floor(index / gridSize)
                    let col = index % gridSize
                    x = col * tileSize
                    y = row * tileSize
                }
            }
        }
    }
}
