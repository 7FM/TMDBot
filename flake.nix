{
  description = "TMDBot env flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils}:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };

      myPyPackages = python-packages: with python-packages; [
        python-telegram-bot
        pyyaml
        (buildPythonPackage rec {
          pname = "tmdbv3api";
          version = "1.9.0";

          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-UExdprmcRRb/FgoBV2ES0JfyCcBTT5Q8FcS1bL2Swzs=";
          };

          propagatedBuildInputs = [
            requests
            urllib3
          ];

          doCheck = false;

          pythonImportsCheck = [ "tmdbv3api" ];
        })
      ];

      tmdbot = pkgs.callPackage ({ python3 }: python3.pkgs.buildPythonApplication {
          pname = "tmdbot";
          version = "0.1";
          src = ./.;
          propagatedBuildInputs = myPyPackages python3.pkgs;
          doCheck = false;
      }) {};

      myPythonWithPackages = pkgs.python3.withPackages (python-packages: (myPyPackages python-packages) ++ (with python-packages; [
        autopep8 # autoformatter
      ]));
    in {
      packages = {
        inherit tmdbot;
        default = tmdbot;
      };
      # TODO use stdEnv?
      devShell = pkgs.mkShellNoCC {
        nativeBuildInputs = [
          # Python env for the utility scripts & cocotb
          myPythonWithPackages
        ];
      };
    }
  );
}
